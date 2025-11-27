from groq import Groq
from dotenv import dotenv_values
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_vars = dotenv_values(os.path.join(BASE_DIR, ".env"))
GroqAPIKey = env_vars["GroqAPIKey"]

client = Groq(api_key=GroqAPIKey)

funcs = [
    "exit", "general", "realtime", "open", "close", "play",
    "generate image", "system", "content", "google search",
    "youtube search", "reminder"
]

ChatHistory = [
    # GENERAL
    {"role": "user", "content": "how are you"},
    {"role": "assistant", "content": "general how are you"},

    {"role": "user", "content": "do you like pizza"},
    {"role": "assistant", "content": "general do you like pizza"},

    # OPEN
    {"role": "user", "content": "open chrome"},
    {"role": "assistant", "content": "open chrome"},

    {"role": "user", "content": "open notepad"},
    {"role": "assistant", "content": "open notepad"},

    # CLOSE
    {"role": "user", "content": "close chrome"},
    {"role": "assistant", "content": "close chrome"},

    # GOOGLE SEARCH
    {"role": "user", "content": "google who is ms dhoni"},
    {"role": "assistant", "content": "google search who is ms dhoni"},

    # YOUTUBE SEARCH
    {"role": "user", "content": "search arjit singh on youtube"},
    {"role": "assistant", "content": "youtube search arjit singh"},

    # PLAY MUSIC
    {"role": "user", "content": "play kesariya song"},
    {"role": "assistant", "content": "play kesariya song"},

    # SYSTEM
    {"role": "user", "content": "increase volume"},
    {"role": "assistant", "content": "system volume up"},

    {"role": "user", "content": "mute the volume"},
    {"role": "assistant", "content": "system mute"},

    # REMINDER
    {"role": "user", "content": "remind me to study at 8 pm"},
    {"role": "assistant", "content": "reminder 8 pm study"},

    # IMAGE GENERATION
    {"role": "user", "content": "generate a photo of a sunset mountain"},
    {"role": "assistant", "content": "generate image sunset mountain"},

    # WEATHER (NEW)
    {"role": "user", "content": "what is temperature now"},
    {"role": "assistant", "content": "realtime what is temperature now"},

    {"role": "user", "content": "is today hot"},
    {"role": "assistant", "content": "realtime is today hot"},

    {"role": "user", "content": "will it rain today"},
    {"role": "assistant", "content": "realtime will it rain today"},

    {"role": "user", "content": "how is the weather outside"},
    {"role": "assistant", "content": "realtime how is the weather outside"},

    # MULTIPLE COMMANDS
    {"role": "user", "content": "open chrome and search for mahadev quotes"},
    {"role": "assistant", "content": "open chrome, google search mahadev quotes"},

    #top news
    {"role": "user", "content": "tell me today's news"},
    {"role": "assistant", "content": "realtime today's news"},

    {"role": "user", "content": "top news today"},
    {"role": "assistant", "content": "realtime top news today"},

    {"role": "user", "content": "breaking news right now"},
    {"role": "assistant", "content": "realtime breaking news right now"},

    {"role": "user", "content": "current headlines"},
    {"role": "assistant", "content": "realtime current headlines"},

    {"role": "user", "content": "what is happening in the world today"},
    {"role": "assistant", "content": "realtime what is happening in the world today"},

]

preamble = """
You are a Decision-Making Model (DMM). You NEVER answer questions.
You only classify queries into the correct task type.

Rules:
- "open X" → open
- "close X" → close
- "play X" → play
- "google ..." → google search
- "youtube ..." → youtube search
- "remind me" → reminder
- "generate a picture/photo/image" → generate image
- "increase volume / mute" → system
- "news", "headlines", "breaking news", "top news" → realtime

Realtime queries are:
weather, temperature, humidity, rainfall, forecast, climate, sunrise, sunset,
"is today hot", "is it raining", "how cold is outside", today's date, time.

General queries are:
normal chatting / opinions / history / no automation.

If multiple tasks, output them separated by commas.
If unclear, classify as: general (query).

OUTPUT MUST FOLLOW THIS EXACT FORMAT:
open chrome
google search virat kohli
play kesariya
general who is sachin
realtime what is temperature now
"""

def FirstLayerDMM(prompt: str = "test"):
    stream = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": preamble},
            *ChatHistory,
            {"role": "user", "content": prompt}
        ],
        stream=True,
        temperature=0.2,
        max_tokens=80
    )

    result = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            result += chunk.choices[0].delta.content

    result = result.replace("\n", "")
    parts = [p.strip() for p in result.split(",")]

    filtered = []
    for task in parts:
        for f in funcs:
            if task.startswith(f):
                filtered.append(task)

    if not filtered:
        filtered = [f"general {prompt}"]

    return filtered


if __name__ == "__main__":
    while True:
        q = input(">>> ")
        print(FirstLayerDMM(q))
