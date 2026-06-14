"""Tests for groovebot.style.mood_mapping (MTG-Jamendo tag → 6 classes)."""
from __future__ import annotations

from groovebot.style.mood_mapping import (
    AMBIGUOUS_TAGS,
    DROPPED_TAGS,
    DROPPED_THEME_TAGS,
    MTG_MOODTHEME_TAGS,
    TAG_TO_MOOD,
    coverage_check,
    resolve_clip_moods,
)
from groovebot.style.model import MOODS


def test_every_official_tag_is_either_mapped_or_dropped():
    uncovered, unknown = coverage_check()
    assert uncovered == set(), f"uncovered MTG tags: {sorted(uncovered)}"
    assert unknown == set(), f"tags not in MTG list: {sorted(unknown)}"


def test_all_mood_classes_are_in_moods_vocabulary():
    bad = {v for v in TAG_TO_MOOD.values() if v not in MOODS}
    assert not bad, f"mapping uses unknown mood classes: {bad}"


def test_theme_and_ambiguous_sets_are_disjoint_from_mapping():
    overlap = set(TAG_TO_MOOD) & DROPPED_TAGS
    assert not overlap, f"tag in both mapping and dropped: {overlap}"


def test_dropped_themes_subset_of_dropped_tags():
    assert DROPPED_THEME_TAGS <= DROPPED_TAGS
    assert AMBIGUOUS_TAGS <= DROPPED_TAGS
    assert DROPPED_THEME_TAGS | AMBIGUOUS_TAGS == DROPPED_TAGS


def test_resolve_single_mood_tag():
    assert resolve_clip_moods(["happy"]) == "happy"
    assert resolve_clip_moods(["epic", "motivational"]) == "epic"


def test_resolve_drops_on_disagreement_by_default():
    assert resolve_clip_moods(["happy", "sad"]) is None


def test_resolve_first_match_breaks_ties_by_moods_order():
    out = resolve_clip_moods(["happy", "sad"], rule="first_match")
    assert out in {"happy", "sad"}
    first = next(m for m in MOODS if m in {"happy", "sad"})
    assert out == first


def test_resolve_with_only_dropped_tags_returns_none():
    # "love" is theme, "cool" is ambiguous, "powerful" is ambiguous
    assert resolve_clip_moods(["love", "cool", "powerful"]) is None


def test_resolve_with_only_unknown_tags_returns_none():
    assert resolve_clip_moods(["totally_made_up_tag", "another_one"]) is None


def test_mtg_moodtheme_tags_is_canonical_size():
    # Snapshot guard — if MTG bumps the tag count, this test fails and
    # the mapping file must be revisited.
    assert len(MTG_MOODTHEME_TAGS) == 59
    assert len(set(MTG_MOODTHEME_TAGS)) == 59  # no duplicates
