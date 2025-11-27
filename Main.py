from Frontend.GUI import (
    GraphicalUserInterface,
    SetAsssistantStatus,
    ShowTextToScreen,
    TempDirectoryPath,
    SetMicrophoneStatus,
    AnswerModifier,
    QueryModifier,
    GetMicrophoneStatus,
    GetAssistantStatus,
)

from Backend.Model import FirstLayerDMM
from Backend.RealtimeSearchEngine import RealtimeSearchEngine
from Backend.Automation import Automation
from Backend.SpeechToText import SpeechRecognition
from Backend.Chatbot import ChatBot
from Backend.TextToSpeech import TextToSpeech
from dotenv import dotenv_values

from asyncio import run
from time import sleep
import subprocess
import threading
import json
import os
import time

# ----------------------
# LOAD .env
# ----------------------
env_vars = dotenv_values(".env")
Username = env_vars.get("Username", "User")
Assistantname = env_vars.get("Assistantname", "Assistant")
# ------------------------------------------
# DEVELOPMENT MODE (NO Groq API, NO DMM, NO Chatbot)
# ------------------------------------------
DEVELOPMENT_MODE = False # <<< SET THIS TO FALSE WHEN USING GROQ AGAIN

if DEVELOPMENT_MODE:
    print("\n===== DEVELOPMENT MODE ACTIVE =====")
    print("Groq API is DISABLED. No rate limits. No token usage.")
    print("Use the terminal to test your automations.\n")
    print("Examples:")
    print("  open chrome")
    print("  close notepad")
    print("  play kesariya")
    print("  google search virat kohli")
    print("  youtube search arijit singh songs")
    print("  system mute")
    print("  exit\n")
    print("=================================\n")

    from asyncio import run
    from Backend.Automation import Automation
    import os

    while True:
        cmd = input(">>> ").strip().lower()
        if cmd in ["exit", "quit"]:
            print("Exiting Development Mode...")
            os._exit(0)

        if cmd == "":
            continue

        try:
            run(Automation([cmd]))
        except Exception as e:
            print("Error executing command:", e)

    os._exit(0)

DefaultMessage = f""" {Username}: Hello {Assistantname}, How are you?
{Assistantname}: Welcome {Username}. I am doing well. How may I help you? """

functions = ["open", "close", "play", "system", "content", "google search", "youtube search"]
subprocess_list = []

# -------------------------------------------------------------------
# RATE-LIMIT COOLDOWN HANDLER
# -------------------------------------------------------------------
def HandleRateLimit():
    cooldown = 60  # seconds
    print(f"[Groq] 429 Rate Limit — Cooling down for {cooldown} seconds...")
    SetAsssistantStatus("Rate limit reached! Cooling down…")
    SetMicrophoneStatus("False")  # Stop auto-execution during cooldown

    for i in range(cooldown):
        print(f"Cooldown: {cooldown - i} sec remaining", end="\r")
        time.sleep(1)

    print("\nCooldown complete.")
    SetAsssistantStatus("Available...")
    SetMicrophoneStatus("True")   # Resume
    return False

# -------------------------------------------------------------------
# SHOW DEFAULT CHAT IF BLANK
# -------------------------------------------------------------------
def ShowDefaultChatIfNoChats():
    try:
        with open(r'Data\ChatLog.json', "r", encoding='utf-8') as file:
            if len(file.read()) < 5:
                with open(TempDirectoryPath('Database.data'), 'w', encoding='utf-8') as temp_file:
                    temp_file.write("")
                with open(TempDirectoryPath('Responses.data'), 'w', encoding='utf-8') as response_file:
                    response_file.write(DefaultMessage)
    except FileNotFoundError:
        print("ChatLog.json not found — creating new one.")
        os.makedirs("Data", exist_ok=True)
        with open(r'Data\ChatLog.json', "w", encoding='utf-8') as file:
            file.write("[]")
        with open(TempDirectoryPath('Responses.data'), 'w', encoding='utf-8') as response_file:
            response_file.write(DefaultMessage)


def ReadChatLogJson():
    try:
        with open(r'Data\ChatLog.json', 'r', encoding='utf-8') as file:
            return json.load(file)
    except:
        return []


def ChatLogIntegration():
    data = ReadChatLogJson()
    formatted = ""

    for entry in data:
        if entry["role"] == "user":
            formatted += f"{Username}: {entry['content']}\n"
        elif entry["role"] == "assistant":
            formatted += f"{Assistantname}: {entry['content']}\n"

    os.makedirs(TempDirectoryPath(''), exist_ok=True)

    with open(TempDirectoryPath('Database.data'), 'w', encoding='utf-8') as file:
        file.write(AnswerModifier(formatted))


def ShowChatOnGUI():
    try:
        with open(TempDirectoryPath('Database.data'), 'r', encoding='utf-8') as file:
            data = file.read()
        if data.strip():
            with open(TempDirectoryPath('Responses.data'), 'w', encoding='utf-8') as r:
                r.write(data)
    except:
        pass


# -------------------------------------------------------------------
# INITIAL SETUP
# -------------------------------------------------------------------
def InitialExecution():
    SetMicrophoneStatus("False")
    ShowTextToScreen("")
    ShowDefaultChatIfNoChats()
    ChatLogIntegration()
    ShowChatOnGUI()


# -------------------------------------------------------------------
# MAIN EXECUTION LOGIC
# -------------------------------------------------------------------
def MainExecution():
    try:
        TaskExecution = False
        ImageExecution = False
        ImageGenerationQuery = ""

        SetAsssistantStatus("Listening...")
        Query = SpeechRecognition()

        # ----------------------------------
        # DOT / EMPTY PROTECTION
        # ----------------------------------
        bad_inputs = ["", ".", "..", "...", ",", "?", " "]
        if Query.strip() in bad_inputs:
            print("[Ignored empty speech input]")
            SetAsssistantStatus("Available...")
            return False

        ShowTextToScreen(f"{Username}: {Query}")
        SetAsssistantStatus("Thinking...")

        # ----------------------------------
        # DECISION MODEL
        # ----------------------------------
        try:
            Decision = FirstLayerDMM(Query)
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                return HandleRateLimit()
            print(f"Decision error: {e}")
            return False

        print("\nDecision:", Decision, "\n")

        G = any(i.startswith("general") for i in Decision)
        R = any(i.startswith("realtime") for i in Decision)

        Merged_query = " and ".join([" ".join(i.split()[1:]) for i in Decision if i.startswith(("general", "realtime"))])

        # ----------------------------------
        # TASK EXECUTION
        # ----------------------------------
        for q in Decision:
            if not TaskExecution and any(q.startswith(func) for func in functions):
                run(Automation(list(Decision)))
                TaskExecution = True

        # ----------------------------------
        # GENERAL CHAT
        # ----------------------------------
        if G:
            try:
                Answer = ChatBot(QueryModifier(Merged_query))
            except Exception as e:
                if "429" in str(e):
                    return HandleRateLimit()
                raise e

            ShowTextToScreen(f"{Assistantname}: {Answer}")
            TextToSpeech(Answer)
            return True

        # ----------------------------------
        # REAL-TIME SEARCH
        # ----------------------------------
        if R:
            try:
                Answer = RealtimeSearchEngine(QueryModifier(Merged_query))
            except Exception as e:
                if "429" in str(e):
                    return HandleRateLimit()
                raise e

            ShowTextToScreen(f"{Assistantname}: {Answer}")
            TextToSpeech(Answer)
            return True

    except Exception as e:
        print("Error in MainExecution:", e)
        return False


# -------------------------------------------------------------------
# THREAD 1 — MAIN LOOP
# -------------------------------------------------------------------
def FirstThread():
    while True:
        try:
            Status = GetMicrophoneStatus()

            if Status.lower() == "true":
                MainExecution()
                sleep(0.25)

            else:
                if "Available..." not in GetAssistantStatus():
                    SetAsssistantStatus("Available...")

            sleep(0.1)

        except Exception as e:
            print("Error in FirstThread:", e)
            sleep(1)


# -------------------------------------------------------------------
# THREAD 2 — GUI LOOP
# -------------------------------------------------------------------
def SecondThread():
    try:
        GraphicalUserInterface()
    except Exception as e:
        print("GUI Error:", e)


# -------------------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------------------
if __name__ == "__main__":
    InitialExecution()

    thread1 = threading.Thread(target=FirstThread, daemon=True)
    thread1.start()

    SecondThread()
