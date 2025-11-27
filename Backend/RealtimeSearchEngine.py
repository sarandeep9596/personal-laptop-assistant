from googlesearch import search
from groq import Groq
from json import load, dump
import datetime
from dotenv import dotenv_values
import requests
import geocoder
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_vars = dotenv_values(os.path.join(BASE_DIR, ".env"))

Username = env_vars.get("Username")
Assistantname = env_vars.get("Assistantname")
GroqAPIKey = env_vars.get("GroqAPIKey")
NEWS_API_KEY = env_vars.get("NEWSAPI")  # <-- add in .env: NEWSAPI = your_key

client = Groq(api_key=GroqAPIKey)
CHATLOG_PATH = os.path.join(BASE_DIR, "Data", "ChatLog.json")

weather_map = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Foggy", 48: "Rime fog", 51: "Light drizzle", 61: "Rain",
    71: "Snow", 95: "Thunderstorm"
}


# --------------------------- TIME & DATE ---------------------------
def _get_time_date(q):
    now = datetime.datetime.now()
    if "time" in q:
        return f"The current time is {now.strftime('%I:%M %p')}."
    if "date" in q:
        return f"Today's date is {now.strftime('%d %B %Y, %A')}."
    if "day" in q:
        return f"Today is {now.strftime('%A')}."
    return None


# --------------------------- WEATHER -------------------------------
def _get_weather(q):
    try:
        g = geocoder.ip("me")
        lat, lon = g.latlng

        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&hourly=temperature_2m,"
            f"precipitation_probability,weathercode&current_weather=true"
        )
        data = requests.get(url).json()

        temp = data["current_weather"]["temperature"]
        code = data["current_weather"]["weathercode"]
        cond = weather_map.get(code, "Unknown weather")

        # FIX: protect against missing index
        rain_data = data["hourly"]["precipitation_probability"]
        rain = rain_data[0] if rain_data else 0

        if "rain" in q:
            return f"There is {rain}% chance of rain today. {'Carry an umbrella 🌧' if rain > 50 else 'No umbrella needed ☀'}"

        if any(w in q for w in ["hot", "cold", "temperature", "degree"]):
            return f"The temperature is {temp}°C and the weather is {cond}."

        if "weather" in q:
            return f"Currently it is {cond} with {temp}°C temperature and {rain}% chance of rain."

        # fallback general weather reply
        return f"It is {cond} with {temp}°C temperature."
    
    except Exception as e:
        return "Unable to fetch weather details at the moment."

# --------------------------- NEWS -------------------------------
def NewsAPI():
    from dotenv import dotenv_values
    import requests
    import os

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_vars = dotenv_values(os.path.join(BASE_DIR, ".env"))
    API_KEY = env_vars.get("GNEWS_API")

    url = f"https://gnews.io/api/v4/top-headlines?category=general&lang=en&country=in&max=5&apikey={API_KEY}"
    res = requests.get(url).json()
    articles = res.get("articles", [])

    news = "Here are the top news today:\n[start]\n"
    for i, a in enumerate(articles):
        news += f"{i+1}. {a['title']}\n"
        if a.get("description"):
            news += f" → {a['description']}\n\n"
    news += "[end]"
    return news

# --------------------------- GOOGLE SEARCH -------------------------------
def GoogleSearch(query):
    results = list(search(query, advanced=True, num_results=5))
    ans = f"Here are the top results for '{query}':\n\n"
    for r in results:
        ans += f"• {r.title}\n{r.description}\n\n"
    return ans


# --------------------------- CLEANING -------------------------------
def AnswerModifier(text):
    return "\n".join([line for line in text.split("\n") if line.strip()])


# --------------------------- MAIN ENGINE -------------------------------
def RealtimeSearchEngine(prompt):
    prompt = prompt.lower().strip()

    # FAST local responses — NO GROQ COST
    if any(w in prompt for w in ["time", "date", "day"]):
        return _get_time_date(prompt)

    if any(w in prompt for w in ["weather", "rain", "cold", "hot", "temperature"]):
        result = _get_weather(prompt)
        return result if result else "Sorry, I couldn't detect the weather."


    if any(w in prompt for w in ["news", "headlines", "breaking", "top news", "today news"]):
        return NewsAPI()

    # --------- Otherwise -> full Groq reasoning + Google ----------
    google_data = {"role": "system", "content": GoogleSearch(prompt)}
    messages = []

    try:
        messages = load(open(CHATLOG_PATH))
    except:
        pass

    messages.append({"role": "user", "content": prompt})

    stream = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[google_data] + messages,
        max_tokens=2048,
        stream=True
    )

    answer = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            answer += chunk.choices[0].delta.content

    answer = AnswerModifier(answer.replace("</s>", ""))
    messages.append({"role": "assistant", "content": answer})

    dump(messages, open(CHATLOG_PATH, "w"), indent=4)
    return answer
