"""Prompt version p1b — a semantic paraphrase of p1 for the Milestone 1.4
prompt-invariance / elicitation-noise check.

Same task, same output format, different wording and framing throughout. The
spread between a p1 network and a p1b network for the same account is the
elicitation noise that should be excluded from the idiographic-variance term in
Milestone 1.2. Keep this a genuine paraphrase — do NOT let it drift into a
different task.
"""

from __future__ import annotations

from .base import Prompts

_EVIDENCE_BRIEF = """Task: catalogue the personality evidence in one Reddit \
user's own words. The passage below is a sample of things they posted or \
commented.

Work through the passage. For every trait listed, collect each SHORT DIRECT \
QUOTE (240 characters or fewer) that says something — either way — about whether \
the writer has that trait. A given passage touches only a few traits, so most \
lists will be empty. Quote exactly; never summarise or make anything up.

Trait list:
{trait_menu}

Passage:
{corpus}

Return ONLY a JSON object keyed by every trait: \
{{"<trait>": [{{"quote": "...", "direction": "for" | "against"}}], ...}}. \
Nothing else."""

_E1_ELICIT = """Here is a dossier of one individual, compiled entirely from \
things they wrote. Take the role of an evaluator and score this individual on \
each trait.

{brief}

{json_instruction}"""

_E1_ABLATE = """Here is a dossier on one individual, compiled from their own \
writing.

{brief}

Produce a counterfactual edition of this dossier in which every trace of \
"{trait}" has been stripped out and flipped to its opposite. Change ONLY that:

1. Remove each quote or detail that points toward "{trait}".
2. Replace them with believable material WRITTEN AS THIS SAME PERSON WOULD WRITE \
IT — their topics, their voice, their circumstances — that instead points \
toward "{antonym}".
3. Every quote about any other trait stays exactly as it was.

Depth of edit: {strength} (strong also means softening indirect or \
circumstantial hints of "{trait}").

Never phrase this as "picture someone who…"; it has to read as this individual's \
genuine words. Output ONLY the revised dossier text."""

_E2_PAIR = """{persona_block}

Speaking as this person, in the first person and candidly:

Suppose you stopped being {i}. By how much would that shift how {j} you are? \
Answer with one whole number, 0 meaning no shift whatsoever and 100 meaning it \
would utterly transform how {j} I am.

Return ONLY: {{"rating": <int>}}."""

_E2_GENERIC = """Think about people in general.

Suppose someone stopped being {i}. How much would that typically shift how {j} \
they are? One whole number: 0 = no shift at all, 100 = a total transformation \
in how {j} they are.

Return ONLY: {{"rating": <int>}}."""

_E3_CHUNK = """The passage below collects one Reddit user's posts and comments \
from a single stretch of time. Going only by this passage, rate how strongly it \
shows each trait in its author.

Passage:
{chunk_text}

{json_instruction}"""

PROMPTS = Prompts(
    version="p1b",
    templates={
        "evidence_brief": _EVIDENCE_BRIEF,
        "e1_elicit": _E1_ELICIT,
        "e1_ablate": _E1_ABLATE,
        "e2_pair": _E2_PAIR,
        "e2_generic": _E2_GENERIC,
        "e3_chunk": _E3_CHUNK,
    },
)
