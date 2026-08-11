"""Detect sentences mangled by extraction or tokenization, before they cost money.

Every rule here was calibrated against the 150-sentence pool rather than guessed, and the
calibration mattered: a naive "bare lowercase letter means a split word" rule would have
deleted `400 m` and `15 m lartësi`, where `m` is the metre unit. The rules below keep
those and still catch `të ardhshë m`.

The filter is deliberately conservative. A false negative is cheap — annotators have a
`junk` flag and will catch what slips through. A false positive is expensive and
invisible: a perfectly good sentence silently never reaches anyone.
"""

from __future__ import annotations

import re

# Albanian single-letter words: articles, particles and clitics that are genuinely one
# character. Anything else standing alone in lowercase is a split-word candidate.
ALBANIAN_ONE_LETTER = {"i", "e", "u", "a", "ë", "o"}

BRACKET_PAIRS = [("(", ")"), ("[", "]"), ("«", "»")]

MAX_DIGIT_RATIO = 0.5
MIN_ALPHA_TOKENS = 4


# Punctuation that stanza sometimes leaves welded to a word (`(Theron)`, `"Perla`).
# Apostrophes are deliberately absent: Albanian contractions (`t'`, `s'`, `n'`) and
# borrowed names (`d'Azur`, `L'`, `Rubin's`) legitimately end or begin with one, and
# splitting those would corrupt real tokens to fix a cosmetic one.
OPENING_PUNCT = '([{"«“'
CLOSING_PUNCT = ')]}"»”'


def repair_tokens(tokens: list[str]) -> list[str]:
    """Peel bracket and quote characters off the edges of word tokens.

    Stanza usually separates punctuation but not always, and a welded token is not a
    cosmetic problem: the entity span becomes `(Theron)` instead of `Theron`, for the
    model *and* for the human annotator, because both work on the same token sequence.
    Neither can express the correct boundary if the tokenizer never offered it.
    """
    out: list[str] = []
    for token in tokens:
        leading: list[str] = []
        trailing: list[str] = []

        while len(token) > 1 and token[0] in OPENING_PUNCT and token[1].isalnum():
            leading.append(token[0])
            token = token[1:]
        while len(token) > 1 and token[-1] in CLOSING_PUNCT and token[-2].isalnum():
            trailing.append(token[-1])
            token = token[:-1]

        out.extend(leading)
        out.append(token)
        out.extend(reversed(trailing))
    return out


def _is_number(token: str) -> bool:
    """Digit-led numerics, including ranges, times and decimals.

    Deliberately permissive about the separators: `1-2`, `12:00` and `2,615` all need to
    count, because each one appeared before a unit letter in the pool and a stricter
    pattern flagged the unit as a split word.
    """
    return bool(re.fullmatch(r"\d[\d.,:\-–/]*", token))


def orphan_letters(tokens: list[str]) -> list[int]:
    """Indices of single letters that look like the tail of a split word.

    Three exemptions, each earned from a real sentence in the pool:

    - **Uppercase** — Roman numerals (`Filip IV`, `Henrit të I`), initials, and labels
      (`A , B , AB dhe 0`).
    - **Adjacent to a number** — a unit or dimension symbol. Either side counts:
      `400 m` and `1-2 m` put the number before, `2,615 milimetra x 3` puts it after.
    - **Between two capitalized tokens** — a particle inside a foreign name
      (`Silva y Velázquez`). Without this the filter deletes a good sentence carrying a
      PER entity, which is exactly the invisible loss the whole module is trying to avoid.
    """
    out = []
    for i, tok in enumerate(tokens):
        if len(tok) != 1 or not tok.isalpha() or not tok.islower():
            continue
        if tok in ALBANIAN_ONE_LETTER:
            continue
        near_number = (i > 0 and _is_number(tokens[i - 1])) or (
            i + 1 < len(tokens) and _is_number(tokens[i + 1])
        )
        if near_number:
            continue  # unit or dimension symbol
        prev_cap = i > 0 and tokens[i - 1][:1].isupper()
        next_cap = i + 1 < len(tokens) and tokens[i + 1][:1].isupper()
        if prev_cap and next_cap:
            continue  # particle inside a name
        out.append(i)
    return out


def hyphen_fragments(tokens: list[str]) -> list[int]:
    """Indices of tokens like `x-Bois` — the tail of a hyphenated name split mid-word.

    Real case from the pool: `Dommartin-aux-Bois` arrived as `Dommartin-au` + `x-Bois`.
    """
    return [
        i
        for i, tok in enumerate(tokens)
        if re.fullmatch(r"[a-zëçA-ZËÇ]-[A-ZËÇ].*", tok) and tok[0].islower()
    ]


def unbalanced_brackets(tokens: list[str]) -> bool:
    """True when brackets don't pair up — reliably a truncated extraction."""
    text = " ".join(tokens)
    return any(text.count(left) != text.count(right) for left, right in BRACKET_PAIRS)


def digit_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if any(c.isdigit() for c in t)) / len(tokens)


def alpha_token_count(tokens: list[str]) -> int:
    return sum(1 for t in tokens if any(c.isalpha() for c in t))


def sentence_issues(tokens: list[str]) -> list[str]:
    """All quality problems found, empty when the sentence looks clean."""
    issues: list[str] = []
    if orphan_letters(tokens):
        issues.append("orphan-letter")
    if hyphen_fragments(tokens):
        issues.append("hyphen-fragment")
    if unbalanced_brackets(tokens):
        issues.append("unbalanced-brackets")
    if digit_ratio(tokens) > MAX_DIGIT_RATIO:
        issues.append("digit-heavy")
    if alpha_token_count(tokens) < MIN_ALPHA_TOKENS:
        issues.append("too-few-words")
    return issues


def is_junk(tokens: list[str]) -> bool:
    return bool(sentence_issues(tokens))
