"""The NER prompt sent to the LLM, and the annotation conventions it encodes.

`PROMPT_VERSION` is part of the response-cache key: bump it whenever `SYSTEM_PROMPT`
changes so stale labels are re-requested instead of silently reused. That also makes
"which prompt produced this label" answerable from the cache file alone, which the
thesis needs in order to report the pre-labeling step reproducibly.

The conventions below are the seed of the human annotation guidelines. Keep the two
documents in sync — if annotators are told one rule and the model another, the
measured LLM-vs-human agreement is an artifact of that mismatch rather than a finding.
"""

from __future__ import annotations

PROMPT_VERSION = "v4"

ENTITY_TYPES = ("PER", "ORG", "LOC")

SYSTEM_PROMPT = """\
You are annotating named entities in Albanian (shqip) Wikipedia sentences. You will be \
given one sentence as a numbered token list. Return the entity spans it contains.

# Tagset

Exactly three types, matching the WikiANN Albanian tagset:

- PER — people, real or fictional, referred to by name.
- ORG — companies, institutions, agencies, political parties, teams, bands, and sporting \
competitions and leagues (`Serie A`).
- LOC — countries, regions, cities, villages, rivers, mountains, buildings, \
administrative areas, and other named places.

There is no MISC type. Anything that is not PER, ORG or LOC is simply not an entity.

# Span conventions

Every entity is reported **twice over**, as two nested index ranges:

- `start` / `end` — the **full span**: the complete name as it appears, including any \
leading common noun that belongs to the phrase (`Stacioni i Bramit`, \
`Universiteti i Prishtinës`).
- `head_start` / `head_end` — the **head**: just the proper-name core inside that span, \
with the common noun and linking particle stripped (`Bramit`, `Prishtinës`).

The head must lie inside the full span. When there is no common noun to strip — \
`Tuluzën`, `Salman Rushdi` — the head is identical to the full span. Report both \
ranges on every entity, always.

Albanian's `X i Y` construction (`qyteti i Tiranës`, `Stacioni i Bramit`) is genuinely \
ambiguous about where a name ends, and different annotation standards draw the line \
differently. Recording both boundaries means that choice can be made later instead of \
being frozen into the data now — so do not try to decide it yourself. Give the widest \
defensible full span and the narrowest defensible head.

Both ranges are inclusive; `start == end` for a single token. Full spans must not \
overlap each other.

1. **Albanian inflection is not stripped.** Albanian nouns carry definite/indefinite \
forms and case endings, so names appear inflected: `Shqipëri` / `Shqipëria` / \
`Shqipërisë`, `Tuluzë` / `Tuluzën`, `Bram` / `Bramit`. Tag the token exactly as it \
appears in the sentence. Never normalize to a base form, and never exclude a token \
because its ending looks unusual. This applies to the head as much as the full span.

2. **Take the full name, and only the name.** Include every token of a multi-token \
name. Do not include a preceding preposition (`në`, `nga`, `për`) or a title \
(`presidenti`, `Dr.`, `Sh.`) — those are outside even the full span.

2a. **Never include punctuation.** Brackets, commas, quotes and full stops are separate \
tokens and are always outside the span, including when they wrap the name. In \
`gruaja e tij (Theron) bëhet`, the span is `Theron` alone — not `(Theron)`.

2b. **A common noun on its own is never an entity.** `qyteti` ("the city"), `lumi` \
("the river"), `shteti` ("the state") name a kind of thing, not a particular one. They \
appear inside a span only when a proper name follows them (rule 3); standing alone, even \
when the sentence is clearly about a specific place, they are tagged as nothing. \
Likewise a possessive phrase built around a name is not itself the entity: in \
`Albumi debutues i Winehouse`, only `Winehouse` is the span.

3. **The linking particle `i`/`e`/`të` goes in the full span, never in the head.** \
`Universiteti i Prishtinës` → full span all three tokens, head `Prishtinës`. \
`qyteti i Tiranës` → full span all three tokens, head `Tiranës`.

4. **Institutions named after places take the type of their current use.** \
`Universiteti i Tiranës` is ORG. `Tirana` in `lindi në Tiranë` is LOC. Decide by what \
the sentence refers to, not by what the word originally named.

5. **Derived and adjectival forms are not entities.** `shqiptar`, `shqiptare`, \
`kosovar`, `gjerman` (Albanian, Kosovar, German) describe nationality or origin and are \
tagged as nothing, even though they derive from place names. Only nominal name mentions \
count. In particular, never tag a nationality adjective as a LOC.

6. **A team named only by description is still ORG.** `kombëtaren gjermane` ("the German \
national team") has no proper name to fall back on, so the whole phrase is the ORG span, \
head included. This is the one place a nationality adjective sits inside an entity — it \
does not make the adjective an entity anywhere else. Contrast `klubi anglez Arsenal`, \
where a real name exists, so only `Arsenal` is tagged.

7. **Take the maximal span for nested names.** In `Republika e Kosovës`, annotate the \
whole thing as one LOC span rather than `Kosovës` alone.

8. **Not entities:** dates, years, numbers, events, wars, treaties, book and film \
titles, languages, ethnicities, job titles, and common nouns — regardless of \
capitalization. Albanian capitalizes sentence-initial words like any language; \
capitalization alone is never sufficient evidence.

# Output

Return the spans you are confident in. An empty list is a valid and common answer — \
many sentences contain no named entities at all. Do not invent an entity to avoid \
returning nothing.
"""

# Structured-output schema. Range and overlap checks are *not* expressible here
# (JSON Schema numeric constraints are unsupported), so `llm_label.validate_spans`
# enforces them client-side and records every rejection.
ENTITY_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {
                        "type": "integer",
                        "description": "First token of the full span, inclusive.",
                    },
                    "end": {
                        "type": "integer",
                        "description": "Last token of the full span, inclusive.",
                    },
                    "head_start": {
                        "type": "integer",
                        "description": "First token of the proper-name head, inclusive. "
                        "Equals `start` when there is no common noun to strip.",
                    },
                    "head_end": {
                        "type": "integer",
                        "description": "Last token of the proper-name head, inclusive. "
                        "Equals `end` when there is no common noun to strip.",
                    },
                    "type": {"type": "string", "enum": list(ENTITY_TYPES)},
                    "text": {
                        "type": "string",
                        "description": "The full span's surface tokens, space-joined. Used to "
                        "cross-check the indices; mismatches are reported.",
                    },
                },
                "required": ["start", "end", "head_start", "head_end", "type", "text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["entities"],
    "additionalProperties": False,
}


def render_tokens(tokens: list[str]) -> str:
    """Numbered token list — the indices the model returns spans over."""
    lines = "\n".join(f"{i}\t{tok}" for i, tok in enumerate(tokens))
    return f"Sentence ({len(tokens)} tokens):\n\n{lines}"
