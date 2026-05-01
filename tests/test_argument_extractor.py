import pytest
from Backend.ArgumentExtractor import extract


@pytest.mark.parametrize("intent,clause,expected", [
    # open
    ("open", "open chrome", "chrome"),
    ("open", "please open notepad", "notepad"),
    ("open", "launch firefox for me please", "firefox"),
    # close
    ("close", "close chrome", "chrome"),
    ("close", "shut down spotify", "spotify"),
    ("close", "quit firefox please", "firefox"),
    # play
    ("play", "play kesariya", "kesariya"),
    ("play", "play arijit singh songs", "arijit singh songs"),
    ("play", "play despacito please", "despacito"),
    # google search
    ("google search", "google who is dhoni", "who is dhoni"),
    ("google search", "search for python tutorials", "python tutorials"),
    ("google search", "google for ipl 2026 schedule", "ipl 2026 schedule"),
    # youtube search
    ("youtube search", "youtube arijit singh", "arijit singh"),
    ("youtube search", "search lo-fi music on youtube", "lo-fi music"),
    ("youtube search", "search for ted talks on youtube", "ted talks"),
    # system — known phrases
    ("system", "increase volume", "volume up"),
    ("system", "louder", "volume up"),
    ("system", "decrease volume", "volume down"),
    ("system", "quieter", "volume down"),
    ("system", "mute", "mute"),
    ("system", "mute the volume", "mute"),
    ("system", "unmute", "unmute"),
    # system — unknown phrase falls through
    ("system", "turn off the dishwasher", "unknown"),
    # generate image
    ("generate image", "generate a photo of sunset mountains", "sunset mountains"),
    ("generate image", "create an image of a dragon", "a dragon"),
    ("generate image", "make a picture of a robot", "a robot"),
    # reminder + general + realtime — verbatim passthrough
    ("reminder", "remind me to call mom at 7", "remind me to call mom at 7"),
    ("general", "tell me a joke", "tell me a joke"),
    ("realtime", "what is the temperature", "what is the temperature"),
])
def test_extract(intent, clause, expected):
    assert extract(clause, intent) == expected
