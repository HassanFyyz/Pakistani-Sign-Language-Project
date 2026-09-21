"""
Decompose "plain" (non-acronym) PSL-CFRT words into ordered letter sequences.

Acronym-style words (e.g. 'او پی ڈی' = O.P.D) are excluded: each
space-separated token there is a spelled-out *English* letter name, not a
decomposable Urdu word, so character-splitting them would produce nonsense
labels. They are held out pending the noun->letter-sequence mapping requested
from the original authors.

The mapping is keyed on `spelling_for_labels(...)`, not `canonicalize(...)` --
see canonicalize_words.py for why the two differ and why using the wrong one
silently corrupts letter alignment.
"""
import json
from collections import Counter

import pandas as pd

from canonicalize_words import canonicalize, spelling_for_labels

ACRONYM_LIKE = {
    'آئی ایس پی آر', 'آئی سی سی', 'آئی سی یو', 'آر ایف ٹی', 'او پی ڈی',
    'ایس او پیز', 'ایس ایچ او', 'ایس پی', 'ایس پیز او', 'ایل ایف ٹی',
    'ایم سی بی', 'ایچ ایف اے ڈی', 'ایچ بی ایل', 'سی ایم ایچ', 'سی بی سی',
    'پی سی آر', 'پی سی بی', 'پی ٹی سی ایل', 'ڈی او', 'ڈی ایس پی',
    'ڈی ایچ اے', 'ڈی پی او',
}

# The PSL fingerspelling alphabet used by PSL-CFRT has 38 letters. Our
# character-level decomposition yields 40 distinct codepoints, because three
# hamza/madda forms appear that are not separate letters of the alphabet:
#   آ  (alef with madda)      ؤ  (waw with hamza)      ئ  (yeh with hamza)
# Whether PSL gives these their own handshape, folds them into the base letter,
# or signs them as base+hamza is a domain question for the authors. Left
# un-merged for now so the choice stays visible; see NOTES in README.
HAMZA_FORMS = {'آ', 'ؤ', 'ئ'}


def decompose(word: str) -> list:
    """Ordered list of individual Urdu letters, spaces removed."""
    return list(word.replace(' ', ''))


def build_mapping(csv_path: str = 'hand_keypoints_features.csv'):
    """Returns (mapping, label_spellings, plain, acronyms, canon_words)."""
    df = pd.read_csv(csv_path, usecols=['video_id'])
    raw = sorted(df['video_id'].unique())

    canon_words = sorted({canonicalize(w) for w in raw})
    label_spellings = sorted({spelling_for_labels(w) for w in raw})

    plain = [w for w in label_spellings if w not in ACRONYM_LIKE]
    acronyms = [w for w in label_spellings if w in ACRONYM_LIKE]
    mapping = {w: decompose(w) for w in plain}
    return mapping, label_spellings, plain, acronyms, canon_words


if __name__ == "__main__":
    mapping, label_spellings, plain, acronyms, canon_words = build_mapping()

    letter_counts = Counter()
    for letters in mapping.values():
        letter_counts.update(letters)

    print(f"Unique canonical words (identity):   {len(canon_words)}")
    print(f"Unique label spellings (what was signed): {len(label_spellings)}")
    print(f"  plain (decomposed): {len(plain)}")
    print(f"  acronym-like (held out): {len(acronyms)}")
    print(f"Distinct letters covered: {len(letter_counts)} "
          f"({len(letter_counts) - len(HAMZA_FORMS & set(letter_counts))} excluding hamza forms)")

    singletons = [l for l, c in letter_counts.items() if c == 1]
    for letter, cnt in letter_counts.most_common():
        tag = "  <- hamza form" if letter in HAMZA_FORMS else ("  <- appears once" if cnt == 1 else "")
        print(f"  {letter}  x{cnt}{tag}")
    print(f"\n{len(singletons)} letters appear in exactly one word: {' '.join(singletons)}")

    with open('word_letter_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print("\nWrote word_letter_mapping.json")
