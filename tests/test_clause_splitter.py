import pytest
from Backend.ClauseSplitter import split


@pytest.mark.parametrize("query,expected", [
    ("open chrome", ["open chrome"]),
    ("", []),
    ("   ", []),
    ("open chrome and play kesariya", ["open chrome", "play kesariya"]),
    ("open chrome and play kesariya then mute", ["open chrome", "play kesariya", "mute"]),
    ("open chrome, then mute", ["open chrome", "mute"]),
    ("OPEN CHROME AND PLAY KESARIYA", ["OPEN CHROME", "PLAY KESARIYA"]),
    ("open chrome   and    play kesariya", ["open chrome", "play kesariya"]),
    ("a, b, c, d", ["a", "b", "c", "d"]),
    ("standalone", ["standalone"]),
])
def test_split(query, expected):
    assert split(query) == expected
