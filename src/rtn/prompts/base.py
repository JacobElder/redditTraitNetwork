"""``Prompts`` container shared by all prompt versions.

A ``Prompts`` instance is a frozen bundle of raw template strings plus render
methods. Paraphrase variants for the prompt-invariance check (Milestone 1.4) are
sibling modules that build a ``Prompts`` with different ``templates`` but the
same render logic.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ..traits import TraitVocab


def _trait_menu(vocab: TraitVocab) -> str:
    lines = []
    for name in vocab.names:
        ant = vocab.antonym[name]
        lines.append(f"- {name} (opposite: {ant})")
    return "\n".join(lines)


def _json_instruction(vocab: TraitVocab) -> str:
    return (
        "Respond with ONLY a JSON object mapping every trait name to an integer "
        f"from {int(vocab.scale_min)} to {int(vocab.scale_max)}. "
        f"{int(vocab.scale_min)} = the trait does not describe this person at "
        f"all; {int(vocab.scale_max)} = it describes them as strongly as it "
        "possibly could. Use null only if there is genuinely no relevant "
        "evidence. No prose, no code fences."
    )


@dataclass(frozen=True)
class Prompts:
    version: str
    templates: Mapping[str, str]

    # -- evidence brief ------------------------------------------------
    def evidence_brief(self, chunk_texts: list[str], vocab: TraitVocab) -> str:
        corpus = "\n\n---\n\n".join(
            f"[chunk {i}]\n{t}" for i, t in enumerate(chunk_texts)
        )
        return self.templates["evidence_brief"].format(
            trait_menu=_trait_menu(vocab),
            corpus=corpus,
            quotes_per_trait="{quotes_per_trait}",
        )

    def evidence_brief_rendered(
        self, chunk_texts: list[str], vocab: TraitVocab, quotes_per_trait: int
    ) -> str:
        return self.evidence_brief(chunk_texts, vocab).replace(
            "{quotes_per_trait}", str(quotes_per_trait)
        )

    # -- E1 -----------------------------------------------------------
    def e1_elicit(self, brief_text: str, vocab: TraitVocab) -> str:
        return self.templates["e1_elicit"].format(
            brief=brief_text,
            json_instruction=_json_instruction(vocab),
        )

    def e1_ablate(
        self, brief_text: str, trait: str, antonym: str, strength: str
    ) -> str:
        return self.templates["e1_ablate"].format(
            brief=brief_text, trait=trait, antonym=antonym, strength=strength
        )

    # -- E2 ---------------------------------------------------------
    def e2_pair(self, i: str, j: str, persona_block: str) -> str:
        return self.templates["e2_pair"].format(
            i=i, j=j, persona_block=persona_block
        )

    def e2_generic(self, i: str, j: str) -> str:
        return self.templates["e2_generic"].format(i=i, j=j)

    # -- E3 -------------------------------------------------------
    def e3_chunk(self, chunk_text: str, vocab: TraitVocab) -> str:
        return self.templates["e3_chunk"].format(
            chunk_text=chunk_text,
            json_instruction=_json_instruction(vocab),
        )
