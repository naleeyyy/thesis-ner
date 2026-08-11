"""Registry of NER models benchmarked on WikiANN-sq.

Each model declares an explicit `label_map`: predicted entity type → WikiANN type
(or "O" to drop). Anything in the model's id2label but absent from `label_map`
is also dropped to "O". This is the single place where cross-model tag space
mismatches get reconciled.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    name: str                # short id used in result tables
    hf_repo: str             # huggingface.co repo id
    label_map: dict[str, str] = field(default_factory=dict)


# WikiANN entity types
WIKIANN_TYPES = {"PER", "ORG", "LOC"}


MODELS: list[ModelSpec] = [
    ModelSpec(
        name="kushtrim-mbert-sq",
        hf_repo="Kushtrim/bert-base-multilingual-cased-finetuned-albanian-ner",
        # Trained directly on WikiANN, label names already match.
        label_map={"PER": "PER", "ORG": "ORG", "LOC": "LOC"},
    ),
    ModelSpec(
        name="akdeniz27-mbert-sq",
        hf_repo="akdeniz27/mbert-base-albanian-cased-ner",
        label_map={"PER": "PER", "ORG": "ORG", "LOC": "LOC"},
    ),
    ModelSpec(
        name="davlan-xlmr-hrl",
        hf_repo="Davlan/xlm-roberta-base-ner-hrl",
        # HRL set predicts PER/ORG/LOC/DATE. DATE has no WikiANN counterpart.
        label_map={"PER": "PER", "ORG": "ORG", "LOC": "LOC", "DATE": "O"},
    ),
    ModelSpec(
        name="babelscape-wikineural",
        hf_repo="Babelscape/wikineural-multilingual-ner",
        # CoNLL-style with MISC; MISC has no WikiANN counterpart.
        label_map={"PER": "PER", "ORG": "ORG", "LOC": "LOC", "MISC": "O"},
    ),
]
