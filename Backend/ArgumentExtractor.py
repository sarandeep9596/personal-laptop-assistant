"""Per-intent rule-based extractor: clause + intent → argument string.
Returns 'unknown' for system commands that don't match any known phrase."""
from __future__ import annotations
import re

_LEADING_FILLERS = (
    "please", "can you", "could you", "would you", "kindly",
    "for me", "now", "right now", "go ahead and",
)

_OPEN_VERBS = ("open", "launch", "start", "fire up", "boot up", "boot", "get me", "fire")
_CLOSE_VERBS = ("close", "shut down", "shut", "quit", "terminate", "stop", "kill", "exit")
_PLAY_VERBS = ("play",)
_GOOGLE_LEADERS = ("google", "search the web for", "search the web", "search for", "search", "for", "up", "on google")
_YOUTUBE_LEADERS = ("youtube", "search for", "search", "for", "on youtube", "up")
_IMAGE_LEADERS = (
    "generate a photo of", "generate an image of", "generate a picture of",
    "create a photo of", "create an image of", "create a picture of",
    "make a photo of", "make an image of", "make a picture of",
    "draw a picture of", "draw an image of", "draw",
    "generate", "create", "make",
)

_SYSTEM_LOOKUP = {
    "volume up": {
        "increase volume", "volume up", "louder", "make it louder",
        "turn it up", "crank the volume", "sound up", "raise the volume",
    },
    "volume down": {
        "decrease volume", "volume down", "quieter", "make it quieter",
        "turn it down", "bring the volume down", "lower the volume",
    },
    "mute": {
        "mute", "mute the volume", "silent", "silence", "silence please",
        "mute audio", "hush", "shh",
    },
    "unmute": {
        "unmute", "unmute the volume", "turn volume on", "sound on", "restore audio",
    },
}


def _strip_leaders(text: str, leaders: tuple[str, ...]) -> str:
    out = text.strip().lower()
    changed = True
    while changed:
        changed = False
        for lead in sorted(leaders, key=len, reverse=True):
            pat = r"^" + re.escape(lead) + r"\b[\s,.:;!?-]*"
            new = re.sub(pat, "", out, count=1)
            if new != out:
                out = new
                changed = True
        for filler in _LEADING_FILLERS:
            pat = r"^" + re.escape(filler) + r"\b[\s,.:;!?-]*"
            new = re.sub(pat, "", out, count=1)
            if new != out:
                out = new
                changed = True
    return re.sub(r"\s+", " ", out).strip()


def _strip_trailing_fillers(text: str, extra_trailers: tuple[str, ...] = ()) -> str:
    out = text
    changed = True
    while changed:
        changed = False
        for filler in (*_LEADING_FILLERS, *extra_trailers):
            pat = r"[\s,.:;!?-]*\b" + re.escape(filler) + r"\s*$"
            new = re.sub(pat, "", out, count=1, flags=re.IGNORECASE)
            if new != out:
                out = new
                changed = True
    return out.strip()


def _system_lookup(clause: str) -> str:
    norm = re.sub(r"\s+", " ", clause.strip().lower())
    for canonical, phrases in _SYSTEM_LOOKUP.items():
        if norm in phrases:
            return canonical
    return "unknown"


def extract(clause: str, intent: str) -> str:
    if intent in ("reminder", "general", "realtime"):
        return clause.strip()

    if intent == "system":
        return _system_lookup(clause)

    if intent == "open":
        return _strip_trailing_fillers(_strip_leaders(clause, _OPEN_VERBS))
    if intent == "close":
        return _strip_trailing_fillers(_strip_leaders(clause, _CLOSE_VERBS))
    if intent == "play":
        return _strip_trailing_fillers(_strip_leaders(clause, _PLAY_VERBS))
    if intent == "google search":
        return _strip_trailing_fillers(
            _strip_leaders(clause, _GOOGLE_LEADERS),
            extra_trailers=("on google",),
        )
    if intent == "youtube search":
        return _strip_trailing_fillers(
            _strip_leaders(clause, _YOUTUBE_LEADERS),
            extra_trailers=("on youtube",),
        )
    if intent == "generate image":
        return _strip_trailing_fillers(_strip_leaders(clause, _IMAGE_LEADERS))

    return clause.strip()
