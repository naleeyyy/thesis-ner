# Albanian NER — Annotation Guidelines

**Version 0.3 (draft, pre-pilot).** Rules will change after the pilot round. If something
seems ambiguous while you work, write it down — that note is worth more than a guess.

---

## What you're doing

Marking **people, organizations, and places** in Albanian sentences from Wikipedia.

A model has already suggested labels for most sentences. **Correct them — don't trust
them.** It is wrong often enough that accepting suggestions unread would ruin the data.
Some sentences come with no suggestions at all; that's deliberate, not a bug.

About 50 sentences per batch, roughly an hour. Take breaks — tired mistakes are the ones
nobody catches.

---

## The three labels

| | | |
|---|---|---|
| **PER** | people | `Salman Rushdi` |
| **ORG** | companies, institutions, parties, teams, leagues | `Universiteti i Prishtinës`, `Serie A` |
| **LOC** | countries, cities, regions, rivers, buildings | `Shqipëria`, `Tuluzën` |

There is no fourth label. Anything that isn't one of these is left unmarked.

---

## Mark two boundaries, not one

Albanian's `X i Y` construction is ambiguous about where a name ends, so mark **both**:

```
Stacioni i Bramit ka lidhje me Tuluzën .
└──── full span ──┘             └ full ┘
          └ head ┘              └ head ┘
        LOC                       LOC
```

- **Full span** — the whole phrase, including a leading common noun.
- **Head** (the `HEAD` label) — just the proper name inside it.

**Most entities need only the full span.** `Tuluzën`, `Salman Rushdi` — nothing to strip,
so skip HEAD entirely. Only add HEAD when it would be narrower.

Rule of thumb: widest defensible full span, narrowest defensible head. Don't agonise over
which is "right" — recording both is the point, so the decision can be made later.

---

## The rules

**1. Never strip inflection.** Albanian names change shape by case and definiteness —
sometimes twice in one sentence:

> Besimtarët katolikë nga **Maqedonia e Veriut** … të **Maqedonisë së Veriut** .

Same country, both marked, each exactly as written. Same for `Tuluzën`, `Bramit`,
`Shkupit`.

**2. `i` / `e` / `të` go in the full span, never in the head.**
`Rajonin e Shkupit` → full span all three, head `Shkupit`.

**3. Prepositions, titles and descriptions stay outside — even the full span.**

> **nga** Maqedonia e Veriut · **Duka i** Brabantit · **klubi anglez** Arsenal

`nga` is a preposition, `Duka` is a title, `klubi anglez` is a description. Only
`Maqedonia e Veriut`, `Brabantit`, and `Arsenal` get marked.

**4. Decide ORG vs LOC by what the sentence means.** Two stations in this corpus:

> **Stacioni i Bramit** ka lidhje… → a name (LOC, head `Bramit`)
> **Stacioni hekurudhor në Zveçan** … → a description; only `Zveçan` (LOC)

**5. Nationality and ethnicity words are not entities.**
`shqiptarët`, `maqedonasit`, `kroatët`, `gjerman`, `italiane` — all unmarked. Never tag
one as a LOC, even though they come from place names.

**6. A team named only by description is still ORG.** `kombëtaren gjermane` ("the German
national team") has no proper name to fall back on, so the whole phrase is one ORG span.
This is the only place a nationality word sits inside an entity — it doesn't make the word
an entity anywhere else. Compare `klubi anglez Arsenal`, where a real name exists, so only
`Arsenal` is marked.

---

## Not entities

All of these appear in the corpus and all look like names at a glance:

| | |
|---|---|
| `Vargjet satanike` | a novel — the model called this **LOC** |
| `Besëlidhjen e Re` | a religious text |
| `Dita e Punës` | a holiday |
| `viti 1183`, `1 maj` | dates |
| `Sekretarin e … Thesarit` | a job title |

Most wear the same `X e Y` costume that *does* signal a name elsewhere. The construction
never decides it — the meaning does.

**Capitalization proves nothing.** Albanian capitalizes the first word of every sentence.

---

## Broken sentences

Some text is garbled by the automatic processing — split words, fragments:

```
704 është përmirësuar dukshëm gjatë produktit të ardhshë m
```

(`ardhshë m` is one word split in two.) **Tick `junk` and move on** — don't try to
annotate it. Junk is flagged, not deleted, so the rate stays measurable.

---

## When you're stuck

1. Could you swap it for "someone" / "some organization" / "some place"? If not, it's
   probably not an entity.
2. Mark your best guess and tick **`unsure`**. A flag is worth more than a confident
   wrong answer.

Sentences with **no entities at all** are common and completely valid.

---

## For Krenar — before circulating

- [x] Albanian examples confirmed; real corpus sentences throughout.
- [x] Junk handling decided: flag, don't delete.
- [ ] Verify the analysis of the examples — the Albanian is authentic but the claims about
      it are mine. Check especially that `Duka` reads as a title, and that `maqedonasit` /
      `kroatët` are ethnicities you'd want unmarked.
- [x] Competitions and national teams decided: both ORG (rules 5 and 6). A nationality
      word inside a team's only designation stays inside the ORG span; it is never a LOC
      in its own right, which keeps the tagset comparable with WikiANN.
- [ ] After the pilot: fold in disagreements, bump to v1.0, and update `SYSTEM_PROMPT` in
      `src/annotate/prompt.py` to match. The two must agree, or measured LLM-vs-human
      agreement reflects the mismatch between two rulebooks. Make prompt edits *before*
      the bulk pre-labeling run — a `PROMPT_VERSION` bump invalidates the response cache.
