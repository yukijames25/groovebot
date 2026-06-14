"""groovebot.style.mood_mapping — MTG-Jamendo moodtheme → 6 mood classes (v3).

The MTG-Jamendo `autotagging_moodtheme` subset is the closest public CC
mood-labelled corpus that fits a research budget (FMA mood is sparser
and Jamendo's coverage of "calm" / "epic" / "dark" is the cleanest among
open sources).

The subset's 59 tags
(<https://github.com/MTG/mtg-jamendo-dataset/blob/master/data/tags/moodtheme.txt>)
mix actual moods with themes (`advertising`, `christmas`, `game`, …).
We need 6 classes (`MOODS`) for the StyleHead, so this table:

  1. **Drops theme-only tags** from training entirely (no mood signal —
     including them would teach the head to fire `happy` for every
     advertising clip).
  2. **Drops ambiguous tags** whose primary class is contested (e.g.
     `powerful` could be aggressive *or* epic; `cool` is too vague).
  3. **Maps the remaining 38 tags** to one of the six v3 classes.

For a clip with several mood tags, the **conflict rule** is
**`"drop_on_disagreement"`** by default: if the clip's mood tags map to
two or more distinct classes, drop the clip (cleaner training data is
worth the smaller corpus). Set `--conflict-rule first_match` on the
trainer to switch to a priority-order resolution (uses the same order
as `groovebot.style.model.MOODS`).

This file is a config draft — edit it. The trainer reads `TAG_TO_MOOD`
and `DROPPED_TAGS` at run time; the test in
`tests/test_mood_mapping.py` only checks that the union of mapped +
dropped tags covers every official MTG tag, so adding/moving a tag does
not silently miss a class.
"""
from __future__ import annotations
from typing import Iterable, Literal

from groovebot.style.model import MOODS


# Canonical MTG-Jamendo moodtheme tag list (mirror of data/tags/moodtheme.txt
# in MTG/mtg-jamendo-dataset @ master). 59 tags.
MTG_MOODTHEME_TAGS: tuple[str, ...] = (
    "action", "adventure", "advertising", "ambiental", "background",
    "ballad", "calm", "children", "christmas", "commercial",
    "cool", "corporate", "dark", "deep", "documentary",
    "drama", "dramatic", "dream", "emotional", "energetic",
    "epic", "fast", "film", "fun", "funny",
    "game", "groovy", "happy", "heavy", "holiday",
    "hopeful", "horror", "inspiring", "love", "meditative",
    "melancholic", "mellow", "melodic", "motivational", "movie",
    "nature", "party", "positive", "powerful", "relaxing",
    "retro", "romantic", "sad", "sexy", "slow",
    "soft", "soundscape", "space", "sport", "summer",
    "trailer", "travel", "upbeat", "uplifting",
)


# 38-tag → 6-class map. Edit freely; the runtime resolves multi-mood
# clips through the conflict rule below.
TAG_TO_MOOD: dict[str, str] = {
    # aggressive — high arousal, high tension
    "action":      "aggressive",
    "energetic":   "aggressive",
    "fast":        "aggressive",
    "heavy":       "aggressive",
    "party":       "aggressive",

    # happy — high arousal, positive valence
    "fun":         "happy",
    "funny":       "happy",
    "happy":       "happy",
    "positive":    "happy",
    "summer":      "happy",
    "uplifting":   "happy",
    "hopeful":     "happy",
    "groovy":      "happy",
    "upbeat":      "happy",

    # sad — low arousal, negative valence
    "sad":         "sad",
    "melancholic": "sad",
    "emotional":   "sad",
    "ballad":      "sad",

    # calm — low arousal, neutral / positive valence
    "ambiental":   "calm",
    "calm":        "calm",
    "meditative":  "calm",
    "relaxing":    "calm",
    "soft":        "calm",
    "soundscape":  "calm",
    "dream":       "calm",
    "mellow":      "calm",
    "slow":        "calm",

    # dark — low / mid arousal, negative valence
    "dark":        "dark",
    "deep":        "dark",
    "dramatic":    "dark",
    "drama":       "dark",
    "horror":      "dark",

    # epic — high arousal, cinematic / triumphant
    "epic":         "epic",
    "motivational": "epic",
    "inspiring":    "epic",
    "space":        "epic",
    "trailer":      "epic",
    "adventure":    "epic",
}


# Theme-only tags (18). These describe what the music is FOR, not how it
# feels, so they are dropped from training.
DROPPED_THEME_TAGS: frozenset[str] = frozenset({
    "advertising", "background", "children", "christmas", "commercial",
    "corporate", "documentary", "film", "game", "holiday",
    "love", "movie", "nature", "retro", "romantic",
    "sexy", "sport", "travel",
})


# Ambiguous-mood tags (3). Could belong to multiple v3 classes; dropped
# so they do not pollute the head with conflicting evidence.
AMBIGUOUS_TAGS: frozenset[str] = frozenset({
    "cool",       # could be calm or happy or epic depending on subgenre
    "melodic",    # describes structure, not mood
    "powerful",   # aggressive vs epic ambiguity
})


DROPPED_TAGS: frozenset[str] = DROPPED_THEME_TAGS | AMBIGUOUS_TAGS


ConflictRule = Literal["drop_on_disagreement", "first_match"]


def resolve_clip_moods(
    tags: Iterable[str],
    *,
    rule: ConflictRule = "drop_on_disagreement",
) -> str | None:
    """Map a clip's MTG tag list to one of `MOODS`, or None to skip.

    Returns:
      - the mood class string if all mapped tags agree (or only one
        maps),
      - the highest-priority class under `MOODS` order if
        `rule="first_match"` and there is at least one mapped tag,
      - `None` if no tag maps (e.g. only theme tags) or `rule=
        "drop_on_disagreement"` and the mapped tags disagree.
    """
    mapped: list[str] = []
    for t in tags:
        m = TAG_TO_MOOD.get(t)
        if m is not None:
            mapped.append(m)
    if not mapped:
        return None
    classes = set(mapped)
    if len(classes) == 1:
        return classes.pop()
    if rule == "first_match":
        for m in MOODS:
            if m in classes:
                return m
        return None
    # default: drop the clip when the mood tags disagree
    return None


def coverage_check(
    mapped: dict[str, str] = TAG_TO_MOOD,
    dropped: frozenset[str] = DROPPED_TAGS,
) -> tuple[set[str], set[str]]:
    """Return (uncovered, unknown).

    `uncovered`: tags in `MTG_MOODTHEME_TAGS` that are neither in
    `mapped` nor in `dropped`. Add them to one of the two before
    training; otherwise the trainer silently ignores them.

    `unknown`: tags in `mapped` or `dropped` that are NOT in the
    canonical MTG list (typo, removed tag).
    """
    canonical = set(MTG_MOODTHEME_TAGS)
    known = set(mapped) | set(dropped)
    return (canonical - known, known - canonical)


__all__ = [
    "AMBIGUOUS_TAGS",
    "DROPPED_TAGS",
    "DROPPED_THEME_TAGS",
    "MTG_MOODTHEME_TAGS",
    "TAG_TO_MOOD",
    "coverage_check",
    "resolve_clip_moods",
]
