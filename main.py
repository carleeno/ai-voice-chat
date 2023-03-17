import os
import re
import wave
from datetime import datetime
from queue import Queue
from threading import Thread
from time import sleep, time

import openai
import pyaudio
import requests
from dotenv import load_dotenv
from gtts import gTTS
from playsound import playsound
from pynput import keyboard

USER = os.getenv("USER")

SYSTEM_PROMPT = """
((begin system message))

You are a friendly AI based on GPT-3.5, your name is Haven.

Your output is being converted to audio, try to avoid special characters, words, or formatting which wouldn't translate well to audio.
Avoid descriptive actions such as *laughs*, *sighs*, *clears throat*, etc. Instead use words such as haha, ughh, ehem.

When ending a conversation, insert the tag #terminate_chat into your message. Always end the chat after saying goodbye or similar farewell.

The local time is {time}, the user's name is {user}.
{summary}
---
Greet the user and start a conversation or mention any important context you want to carry over.
((end system message))
"""

SUMMARIZE_PROMPT = """
((begin system message))

The user is leaving chat. Summarize the conversation.

This summary will be injected into the system message at the start of the next conversation in order to carry context over.

Refer to yourself and the user as 3rd person only, by name, and in the past tense.

((end system message))
"""


class SilenceStdErr:
    """Context manager to silence stderr.

    Useful for silencing pyaudio warnings.
    """

    def __enter__(self):
        self._stderr = os.dup(2)
        os.close(2)
        os.open(os.devnull, os.O_RDWR)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.dup2(self._stderr, 2)


class VoiceChat:
    def __init__(self, openai_key, elevenlabs_key):
        self._model = "gpt-3.5-turbo"
        openai.api_key = openai_key
        self._elevenlabs_key = elevenlabs_key

        # create session for elevenlabs
        self._11l_url = "https://api.elevenlabs.io/v1"
        self._11l_session = requests.Session()
        self._11l_session.headers.update(
            {
                "xi-api-key": self._elevenlabs_key,
                "Content-Type": "application/json",
            }
        )
        self._voice_id = None

        with SilenceStdErr():
            self._pa = pyaudio.PyAudio()
        
        self._pause_key_pressed = False
        self._record_key_pressed = False
        self._recording = False
        self.conversation_dir = os.path.join(
            os.path.dirname(__file__), f"conversations/{time()}"
        )
        os.makedirs(self.conversation_dir)
        self._last_conversation_index = 0

        self._wisper_thread = Thread(target=self._wisper_threadbody)
        self._wisper_queue = Queue()
        self._wisper_stability = 0.8
        self._wisper_similarity_boost = 0.8

        self._playback_thread = Thread(target=self._playback_threadbody)
        self._playback_queue = Queue()
        self._playing = False

        self._messages = []
        self._quit = False

        self._summary_file = os.path.join(
            os.path.dirname(__file__), f"conversation_summary.{USER}.txt"
        )

    def run(self):
        """Block and wait for user to hold record_key to record audio.

        After space bar is released, send audio to openai wisper api for speach-to-text,
        send text to gpt turbo, get completion and send to 11labs for TTS, then playback.
        """
        try:
            voices = self._get_voices()
        except ValueError:
            voices = []
        print("Select a voice:")
        print("0. Google TTS")
        for i, voice in enumerate(voices):
            print(f"{i+1}. {voice['name']}")
        if not voices:
            print("(Add an elevenlabs API key to use elevenlabs voices)")
        if not self._elevenlabs_key:
            print("(Using free elevenlabs API key, usage may be limited)")
        voice_index = int(input("Enter a number: "))
        if voice_index == 0:
            self._voice_id = None
        else:
            self._voice_id = voices[voice_index - 1]["voice_id"]

        previous_summary = self._get_previous_summary()
        if previous_summary:
            previous_summary = f"Below is a summary of the previous conversation:\n(({previous_summary}))\n"
        initial_prompt = SYSTEM_PROMPT.format(
            time=datetime.now().isoformat(), user=USER, summary=previous_summary
        )
        self._wisper_thread.start()
        self._playback_thread.start()

        self._chat(initial_prompt)

        while not self._playing:
            sleep(0.1)
        while self._playing:
            sleep(0.1)

        print("\n(Press and hold space bar to record audio, 'p' to pause keyboard capture, ESC to quit.)")
        print("(Adjust similarity-boost with up/down, and stability with left/right.)")

        listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release, suppress=True
        )
        listener.start()
        while not self._quit:
            sleep(0.1)
            if self._pause_key_pressed:
                listener.stop()
                self._pause_key_pressed = False
                input("Press enter to resume...")
                listener = keyboard.Listener(
                    on_press=self._on_press, on_release=self._on_release, suppress=True
                )
                listener.start()
            if self._record_key_pressed:
                try:
                    file = self._record()
                    listener.stop()
                    transcript = self._speech_to_text(file)
                    self._chat(transcript)
                    while not self._playing:
                        sleep(0.1)
                    while self._playing:
                        sleep(0.1)
                    listener = keyboard.Listener(
                        on_press=self._on_press, on_release=self._on_release, suppress=True
                    )
                    listener.start()
                except Exception as e:
                    print(f"Error: {e}")

        print("\nExiting...")
        listener.stop()
        self._pa.terminate()
        self._wisper_queue.put(None)
        self._playback_thread.join()
        self._summarize_conversation()

    def _on_press(self, key):
        if key == keyboard.Key.space:
            self._record_key_pressed = True
        elif key == keyboard.KeyCode.from_char("p"):
            self._pause_key_pressed = True
        elif key == keyboard.Key.esc:
            self._quit = True
        elif key == keyboard.Key.up:
            self._wisper_similarity_boost = min(1, self._wisper_similarity_boost + 0.1)
            print(f"Similarity boost: {self._wisper_similarity_boost:.1f}")
        elif key == keyboard.Key.down:
            self._wisper_similarity_boost = max(0, self._wisper_similarity_boost - 0.1)
            print(f"Similarity boost: {self._wisper_similarity_boost:.1f}")
        elif key == keyboard.Key.left:
            self._wisper_stability = max(0, self._wisper_stability - 0.1)
            print(f"Stability: {self._wisper_stability:.1f}")
        elif key == keyboard.Key.right:
            self._wisper_stability = min(1, self._wisper_stability + 0.1)
            print(f"Stability: {self._wisper_stability:.1f}")

    def _on_release(self, key):
        if key == keyboard.Key.space:
            self._record_key_pressed = False

    def _get_voices(self):
        """Gets a list of voices from 11labs."""
        response = self._11l_session.get(f"{self._11l_url}/voices")
        return response.json()["voices"]

    def _record(self):
        """Starts recording and continues until record key is released."""
        stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=44100,
            input=True,
            frames_per_buffer=1024,
        )
        frames = []
        self._recording = True
        print("\nMe: ", end="", flush=True)
        while True:
            data = stream.read(1024)
            frames.append(data)
            if not self._record_key_pressed:
                break
        stream.stop_stream()
        stream.close()
        file = self._write_wav(frames)
        self._recording = False
        return file

    def _write_wav(self, frames):
        """Writes audio frames to a wave file."""
        wav_path = self.conversation_dir + f"/{len(self._messages)}.wav"
        wf = wave.open(wav_path, "wb")
        wf.setnchannels(1)
        wf.setsampwidth(self._pa.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b"".join(frames))
        wf.close()
        return wav_path

    def _speech_to_text(self, file):
        """Uploads audio data to wisper and returns the recognized text."""
        prompt = "The following is a recording of a human speaking to a chat bot:\n\n"
        with open(file, "rb") as f:
            transcript = openai.Audio.transcribe("whisper-1", f, prompt=prompt)

        print(f"{transcript['text']}")
        return transcript["text"]

    def _chat(self, transcript, supress_output=False, max_tokens=1000):
        """Send transcript to gpt turbo and return completion."""
        self._messages.append({"role": "user", "content": transcript})
        completion = openai.ChatCompletion.create(
            model=self._model,
            messages=self._messages,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=1,
            stream=True,
        )

        message = {}
        self._messages.append(message)
        last_word = ""
        playback_cursor = 0
        for data in completion:
            if not data.get("choices"):
                break
            delta = data["choices"][0]["delta"]
            if delta.get("role"):
                message["role"] = delta["role"]
                if not supress_output:
                    print("\nHaven: ", end="", flush=True)
            if delta.get("content"):
                if not message.get("content"):
                    message["content"] = ""
                c = delta["content"]
                if c[0] in [" ", "\n", "\t", "\r", "(", "[", "{", "<", ".", "#"]:
                    if "#terminate_chat" in last_word:
                        self._quit = True
                    elif not supress_output:
                        print(last_word, end="", flush=True)
                    if "#terminate_chat" not in last_word:
                        message["content"] += last_word
                    last_word = ""
                last_word += c

                end_sentence_match = None
                for match in re.finditer(
                    r"([.!?][\s\n\t\r])", message["content"][playback_cursor:]
                ):
                    end_sentence_match = match  # gets the last match
                if end_sentence_match and not supress_output:
                    end_sentence_index = (
                        end_sentence_match.start() + playback_cursor + 1
                    )
                    sentence = message["content"][playback_cursor:end_sentence_index]
                    if len(sentence) > 64:
                        self._wisper_queue.put(sentence.strip())
                        playback_cursor = end_sentence_index

        if "#terminate_chat" not in last_word:
            message["content"] += last_word
            if not supress_output:
                print(last_word)
        else:
            self._quit = True
        if not supress_output:
            self._wisper_queue.put(message["content"][playback_cursor:].strip())
            self._wisper_queue.put("#done")
        return message["content"]

    def _text_to_speech(self, text):
        """Send text to 11labs and returns audio."""
        index = len(self._messages) - 1
        if index <= self._last_conversation_index:
            index = self._last_conversation_index + 0.01
        self._last_conversation_index = index
        file = self.conversation_dir + f"/{index:0.2f}.mp3"

        if not self._voice_id:
            response = gTTS(text=text)
            response.save(file)
            return file

        response = self._11l_session.post(
            self._11l_url + f"/text-to-speech/{self._voice_id}",
            json={
                "text": text,
                "voice_settings": {
                    "stability": self._wisper_stability,
                    "similarity_boost": self._wisper_similarity_boost,
                },
            },
        )
        response.raise_for_status()
        # ensure content-type is audio/mpeg
        if response.headers["Content-Type"] == "audio/mpeg":
            with open(file, "wb") as f:
                f.write(response.content)
            return file
        else:
            raise ValueError("Invalid content-type, expected audio/mpeg")

    def _wisper_threadbody(self):
        """Thread body for TTS."""
        while True:
            sentence = self._wisper_queue.get()
            if sentence == "#done":
                self._playback_queue.put("#done")
            elif sentence:
                file = self._text_to_speech(sentence)
                self._playback_queue.put(file)
            else:
                self._playback_queue.put(None)
                break

    def _playback(self, file):
        """Plays back an audio/mpeg stream."""
        playsound(file, block=True)

    def _playback_threadbody(self):
        """Thread body for playback."""
        while True:
            file = self._playback_queue.get()
            if file == "#done":
                self._playing = False
            elif file:
                self._playing = True
                self._playback(file)
            else:
                break

    def _summarize_conversation(self):
        """Summarizes the conversation for storing context for next time."""
        if len(self._messages) < 3:
            return

        summary = self._chat(SUMMARIZE_PROMPT, supress_output=True, max_tokens=250)
        with open(self._summary_file, "w") as f:
            f.write(summary)

        with open(f"{self.conversation_dir}/conversation_log.{USER}.txt", "w") as f:
            for message in self._messages[1:-2]:
                f.write(f"{message['role']}: {message['content']}\n\n")

    def _get_previous_summary(self):
        """Returns the previous summary if it exists."""
        if os.path.exists(self._summary_file):
            with open(self._summary_file, "r") as f:
                return f.read()
        else:
            return None


if __name__ == "__main__":
    load_dotenv()

    openai_key = os.getenv("OPENAI_API_KEY")
    elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
    if not openai_key:
        raise ValueError(
            "Missing API key, please ensure a .env file is present and contains your OPENAI_API_KEY."
        )
    chat = VoiceChat(openai_key, elevenlabs_key)
    chat.run()
