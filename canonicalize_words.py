"""
Canonicalize PSL-CFRT video_id (word) strings.

Two distinct problems live in the raw filenames:

  1. Unicode codepoint variants that render identically (Arabic Kaf vs Urdu
     Keheh, Arabic Yeh vs Urdu Yeh). Pure encoding noise -- always safe to fix.
     Handled by CHAR_NORMALIZE.

  2. Genuine transcription inconsistencies between clips (stray spaces,
     alternate spellings, one annotator adding "Bank" where another did not).
     Handled by MANUAL_ALIASES.

IMPORTANT -- these two problems need DIFFERENT treatment depending on what the
caller wants, and conflating them silently corrupts letter-level labels:

  canonicalize(raw)          -> WORD IDENTITY. Use for grouping, class labels,
                                counting unique words. Applies everything.

  spelling_for_labels(raw)   -> WHAT WAS ACTUALLY SPELLED IN THAT CLIP. Use for
                                letter-level decomposition and pseudo-labelling.
                                Applies encoding fixes and length-preserving
                                typo fixes only.

Why the split: four aliases change the number of letters (see
LENGTH_CHANGING_ALIASES). Merging 'سٹینڈرڈ چارٹرڈ بنک' (16 letters) into
'سٹینڈرڈ چارٹرڈ' (13) is right for word identity -- they are the same
real-world entity -- but catastrophic for pseudo-labelling, because the signer
in that clip demonstrably fingerspelled 16 letters. Labelling it with 13 would
misalign every letter after the split point.

After the fixes below: 87 raw filenames -> 72 unique words.
"""
import re
import unicodedata

CHAR_NORMALIZE = {
    'ك': 'ک',  # ARABIC KAF -> URDU KEHEH (ك -> ک)
    'ي': 'ی',  # ARABIC YEH -> URDU YEH (ي -> ی)
    'ة': 'ہ',  # ARABIC TEH MARBUTA -> URDU HEH GOAL (ة -> ہ)
}


def normalize_word(raw: str) -> str:
    """Strip extension, NFC, unify codepoint variants, squeeze whitespace."""
    w = raw.replace('.MP4', '').replace('.mp4', '')
    w = unicodedata.normalize('NFC', w)
    for bad, good in CHAR_NORMALIZE.items():
        w = w.replace(bad, good)
    w = re.sub(r'\s+', ' ', w).strip()
    return w


# Keys here are written as they appear in the raw filenames. They are passed
# through normalize_word() at import time (see _ALIASES) -- writing a key
# containing e.g. ARABIC KAF and looking it up *after* normalization would
# never match, which is a bug this module previously had: two aliases were
# silently dead and the unique-word count read 74 instead of 72.
MANUAL_ALIASES = {
    'جوہر ٹاون': 'جوہر ٹاؤن',
    'سٹینڈرڈ چارٹرڈ بنك': 'سٹینڈرڈ چارٹرڈ',
    'سٹینڈرڈچارٹرڈ': 'سٹینڈرڈ چارٹرڈ',
    'سیٹنڈرڈچارٹرڈ': 'سٹینڈرڈ چارٹرڈ',
    'ماڈل ٹاون': 'ماڈل ٹاؤن',
    'گنگارام': 'گنگا رام',
    'عكسری بنک': 'عسکری بنک',
    'عکسری بنک': 'عسکری بنک',
    'یونائینٹد بنک': 'یونائیٹڈ بنک',
    'یونائیڈ بنک': 'یونائیٹڈ بنک',
    'الائیٹك بنک': 'الائیڈ بنک',
    'الفلا ح بنک': 'الفلاح بنک',
}

# Aliases resolved against already-normalized text.
_ALIASES = {normalize_word(k): normalize_word(v) for k, v in MANUAL_ALIASES.items()}


def _n_letters(w: str) -> int:
    return len(w.replace(' ', ''))


# Aliases that change how many letters get fingerspelled. Safe for word
# identity, NOT safe for letter-level labels. Clips whose recorded spelling
# appears here are flagged rather than silently relabelled.
LENGTH_CHANGING_ALIASES = {
    src: dst for src, dst in _ALIASES.items() if _n_letters(src) != _n_letters(dst)
}

# The subset that is safe to apply when deriving letter sequences.
_SAFE_ALIASES = {
    src: dst for src, dst in _ALIASES.items() if src not in LENGTH_CHANGING_ALIASES
}


def canonicalize(raw: str) -> str:
    """Canonical WORD IDENTITY. Merges spelling variants of the same entity.

    Note: 'سٹینڈرڈ چارٹرڈ بنک' is merged into 'سٹینڈرڈ چارٹرڈ' on the
    assumption that one annotator appended "Bank" to "Standard Chartered".
    That one is genuinely ambiguous and flagged for human review; every other
    alias is an unambiguous typo or spacing fix.
    """
    w = normalize_word(raw)
    return _ALIASES.get(w, w)


def spelling_for_labels(raw: str) -> str:
    """The letter sequence actually fingerspelled in THIS clip.

    Deliberately does not apply length-changing merges -- see module docstring.
    """
    w = normalize_word(raw)
    return _SAFE_ALIASES.get(w, w)


def is_ambiguous_spelling(raw: str) -> bool:
    """True if this clip's recorded spelling disagrees in length with its
    canonical form -- i.e. either the annotator typo'd or the signer really did
    spell something different. Cannot be resolved without watching the video,
    so such clips are excluded from the trusted validation set.
    """
    return normalize_word(raw) in LENGTH_CHANGING_ALIASES
