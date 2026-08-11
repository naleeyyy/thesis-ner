"""Tests for the sentence quality filter.

Every "keeps" case below is a real sentence from the 150-sentence pool that an earlier,
naiver version of this filter deleted. They are regression tests against silently losing
good data — the failure mode that matters here, since a dropped sentence never reaches
an annotator and nothing downstream notices.
"""

from __future__ import annotations

from src.data.quality import (
    alpha_token_count,
    digit_ratio,
    hyphen_fragments,
    is_junk,
    orphan_letters,
    repair_tokens,
    sentence_issues,
    unbalanced_brackets,
)


def toks(s: str) -> list[str]:
    return s.split()


# ------------------------------------------------------------------- orphan letters


def test_catches_a_split_word_tail():
    # The original motivating case: "të ardhshëm" arrived split as "ardhshë" + "m".
    assert orphan_letters(toks("704 është përmirësuar gjatë produktit të ardhshë m")) == [7]


def test_keeps_a_unit_symbol_after_a_number():
    assert orphan_letters(toks("gjendet në lartësi mbidetare 400 m .")) == []
    assert orphan_letters(toks("rritet nga 8 deri në 15 m lartësi .")) == []


def test_keeps_a_particle_inside_a_foreign_name():
    # `Diego Rodríguez de Silva y Velázquez` — deleting this loses a PER entity.
    assert orphan_letters(toks("Diego Rodríguez de Silva y Velázquez ishte piktor")) == []


def test_keeps_albanian_one_letter_words():
    assert orphan_letters(toks("Stacioni i Bramit e ka u bë")) == []


def test_keeps_uppercase_single_letters():
    # Roman numerals, initials and labels are all legitimate here.
    assert orphan_letters(toks("Mbretit Filip IV dhe A , B , AB dhe 0 .")) == []
    assert orphan_letters(toks("Henrit të I të Derës")) == []


def test_a_lowercase_letter_between_lowercase_words_is_still_flagged():
    assert orphan_letters(toks("kjo fjala x ishte ndarë")) == [2]


# ----------------------------------------------------------------- hyphen fragments


def test_catches_a_split_hyphenated_name():
    # `Dommartin-aux-Bois` arrived as `Dommartin-au` + `x-Bois`.
    assert hyphen_fragments(toks("Dommartin-au x-Bois është një komunë në Francë .")) == [1]


def test_keeps_a_normal_hyphenated_word():
    assert hyphen_fragments(toks("piktor gjermaniko-romak dhe indiano-britanez .")) == []


def test_keeps_a_capitalized_hyphenated_name():
    assert hyphen_fragments(toks("Dommartin-Bois është një komunë .")) == []


# ----------------------------------------------------------------------- brackets


def test_catches_unbalanced_brackets():
    assert unbalanced_brackets(toks("ISHK) , është një organ administrativ"))
    assert unbalanced_brackets(toks("A3 , Ax; dhe të fundit A0 dhe A4) ."))


def test_keeps_balanced_brackets():
    assert not unbalanced_brackets(toks("Miguelturra ka 13986 banorë (2009) , dhe përfshinë"))


# ------------------------------------------------------------------ density checks


def test_digit_ratio_keeps_ordinary_geography_sentences():
    # The most digit-dense real sentence in the pool sits at 0.33 and is perfectly usable.
    tokens = toks("Chabottes ka një popullsi prej 801 banorë dhe sipërfaqe 9 .96 km² .")
    assert digit_ratio(tokens) < 0.5
    assert not is_junk(tokens)


def test_digit_heavy_is_flagged():
    assert "digit-heavy" in sentence_issues(toks("1 2 3 4 5 6 viti 7 8 9"))


def test_alpha_token_count():
    assert alpha_token_count(toks("A , B , AB dhe 0 .")) == 4


# ------------------------------------------------------------------------ end to end


def test_clean_sentence_has_no_issues():
    tokens = toks("Stacioni i Bramit ka lidhje hekurudhore me Tuluzën , Karkasonën dhe Narbonën .")
    assert sentence_issues(tokens) == []
    assert not is_junk(tokens)


def test_issues_are_reported_by_name():
    assert sentence_issues(toks("ISHK) , është një organ i Ministrisë")) == ["unbalanced-brackets"]


def test_a_sentence_can_carry_several_issues():
    assert len(sentence_issues(toks("A4) x-Bois 1 2 3"))) > 1


# ---------------------------------- units and dimensions found at 2500-sentence scale


def test_keeps_a_unit_after_a_numeric_range():
    # `e gjatë deri në 1-2 m` — the range broke a stricter number pattern.
    assert orphan_letters(toks("bimë e gjatë deri në 1-2 m , me lule")) == []


def test_keeps_a_unit_after_a_time():
    assert orphan_letters(toks("nga 12:00 , 12:00 m .")) == []


def test_keeps_a_dimension_separator_before_a_number():
    # `2 ,615 milimetra x 3 ...` — x is the multiplication sign, not a split word.
    assert orphan_letters(toks("2 ,615 milimetra x 3 ,048 milimetra")) == []


def test_still_catches_splits_that_merely_sit_near_digits():
    # A real split whose neighbours are words must survive the widened exemption.
    tokens = toks("Bazuar në romanin e vitit 1990 me emër të Andre w")
    assert [tokens[i] for i in orphan_letters(tokens)] == ["w"]


# ------------------------------------------------------------------- deduplication


def test_template_shape_masks_digits():
    from src.data.segment import template_shape

    a = template_shape("Ka sipërfaqe prej 43 .45 km² , lartësi 400 m .")
    b = template_shape("Ka sipërfaqe prej 12 .09 km² , lartësi 87 m .")
    assert a == b


def test_deduplicate_drops_exact_repeats():
    from src.data.segment import deduplicate

    recs = [{"tokens": toks("një dy tre")}, {"tokens": toks("një dy tre")}]
    kept, dropped = deduplicate(recs, max_per_template=3)
    assert len(kept) == 1
    assert dropped[0]["issues"] == ["exact-duplicate"]


def test_deduplicate_caps_a_template_family_without_erasing_it():
    from src.data.segment import deduplicate

    # Genuine negatives are useful; fifteen near-identical copies are not.
    recs = [{"tokens": toks(f"Ka sipërfaqe prej {i} km² sot")} for i in range(15)]
    kept, dropped = deduplicate(recs, max_per_template=3)
    assert len(kept) == 3
    assert len(dropped) == 12
    assert {d["issues"][0] for d in dropped} == {"template-repeat"}


def test_deduplicate_keeps_distinct_sentences():
    from src.data.segment import deduplicate

    recs = [{"tokens": toks(s)} for s in ["një dy tre", "katër pesë gjashtë", "shtatë tetë"]]
    kept, _ = deduplicate(recs, max_per_template=3)
    assert len(kept) == 3


# --------------------------------------------------- token repair (welded punctuation)


def test_repair_splits_wrapping_parentheses():
    # `(Theron)` arrived as one token, so neither model nor annotator could mark `Theron`.
    assert repair_tokens(["gruaja", "(Theron)", "bëhet"]) == ["gruaja", "(", "Theron", ")", "bëhet"]


def test_repair_splits_quotes_around_a_name():
    assert repair_tokens(['"Perla', 'Franceze"']) == ['"', "Perla", "Franceze", '"']


def test_repair_handles_a_year_in_brackets():
    assert repair_tokens(["(2009)"]) == ["(", "2009", ")"]


def test_repair_leaves_albanian_contractions_alone():
    # `t'`, `s'`, `n'` are real Albanian tokens; splitting them corrupts the text.
    for tok in ["t'", "s'", "n'", "t’ja"]:
        assert repair_tokens([tok]) == [tok]


def test_repair_leaves_apostrophe_names_alone():
    for tok in ["d'Azur", "L'", "Rubin's", "Howe's"]:
        assert repair_tokens([tok]) == [tok]


def test_repair_leaves_standalone_punctuation_alone():
    assert repair_tokens([",", "(", ")", '"', "."]) == [",", "(", ")", '"', "."]


def test_repair_leaves_clean_tokens_untouched():
    tokens = ["Stacioni", "i", "Bramit", "ka", "lidhje", "."]
    assert repair_tokens(tokens) == tokens


def test_repair_is_idempotent():
    once = repair_tokens(["(Theron)", '"Perla'])
    assert repair_tokens(once) == once
