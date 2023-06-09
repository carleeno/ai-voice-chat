# ai-voice-chat

A hacky AI voice chat experiment

## Features

- Push-to-talk style input
- Realistic voice TTS using elevenlabs.io
- Conversation keeps its context (just like chatGPT)
- A summary of the previous conversation is saved on exit, to help carry some context over into a new conversation next time you launch it.
- Recordings and text logs of all conversations are saved locally (in ./conversations)

## Problems

- It probably won't work well in languages other than english.
- You can't use your keyboard in other apps while this is running.
- Audio synthesis is slow to generate. This is mitigated some by chunking the response into sentences and starting to synthesize as soon as possible, however this can sometimes create long pauses between sentences if the API response is taking too long. It can also cause the intonation to vary unnaturally from one sentence to the next.
- Inter-conversation context is based on a summary of the previous conversation and thus will give the appearance of poor memory.
- Elevenlabs is expensive, you'll quickly use up your character count while using this. Google TTS is provided as an alternative but sounds terrible in comparison.
- Only tested in Ubuntu, let me know if you get it to work on another OS.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then create a .env file containing `OPENAI_API_KEY=<your_key>` and optionally `ELEVENLABS_API_KEY` as well. You can use elevenlabs voices for free without a key, but will be limited.

Google text to speech sounds horrible but it's an option as well.

## Run

Remember to `source venv/bin/activate` if not already sourced.

`python main.py`

Set your volume ahead of time, it uses pynput to detect when you're holding the space bar to talk, but you can't use your keyboard even for volume.

Hold space bar while you talk, recommend waiting for the AI to finish talking before you talk, it's not possible to cut the AI short yet.

Press ESC to exit.
