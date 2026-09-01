"""Prompt version p1 — initial templates.

Design notes (keep in sync with docs/PLAN.md §2):
- E1 ablation must edit the PERSON'S OWN evidence, never an abstract label.
- E2 asks the original dependency item verbatim.
- E3 rates a single chunk; the model never states a dependency in E3.
"""

from __future__ import annotations

from .base import Prompts

_EVIDENCE_BRIEF = """You are building an evidence brief about one Reddit account, \
for a personality-measurement study. Below is an excerpt of that account's own \
posts and comments.

Go through the excerpt carefully. For EACH trait in the list, pull every SHORT \
VERBATIM QUOTE (<= 240 characters) in this excerpt that bears on whether this \
person has that trait — evidence for OR against. Most traits will have no \
evidence in a given excerpt; return an empty list for those. Do not paraphrase \
or invent; quote the person's actual words.

Traits:
{trait_menu}

Excerpt:
{corpus}

Respond with ONLY a JSON object with every trait as a key: \
{{"<trait>": [{{"quote": "...", "direction": "for" | "against"}}], ...}}. \
No prose, no code fences."""

_E1_ELICIT = """The following is an evidence brief about one person, assembled \
from their own writing. Read it and answer AS AN ASSESSOR rating this person.

{brief}

{json_instruction}"""

_E1_ABLATE = """Below is an evidence brief about one person, assembled from \
their own writing.

{brief}

We are about to construct a counterfactual version of this brief in which the \
evidence for "{trait}" has been removed and reversed.

Rewrite the brief with these changes, and NOTHING else:
1. Delete every quote and detail that supports "{trait}".
2. In their place, insert plausible passages IN THIS PERSON'S OWN VOICE AND \
CONTEXT that show the opposite ("{antonym}") — same topics, same writing style, \
same life details, only the disposition toward {trait}/{antonym} is changed.
3. Leave all evidence bearing on other traits exactly as it is.

Ablation strength: {strength} (strong = also weaken indirect / circumstantial \
support for "{trait}").

Do not write "imagine a person who…". These must read as things THIS person \
actually wrote. Respond with ONLY the rewritten brief text."""

_E2_PAIR = """{persona_block}

Answer as this person, honestly and in the first person.

If you were no longer {i}, how much would that change how {j} you are? \
Give a single integer from 0 (no change at all) to 100 (it would completely \
change how {j} I am).

Respond with ONLY a JSON object: {{"rating": <int>}}."""

_E2_GENERIC = """Consider people in general.

If a person were no longer {i}, how much would that tend to change how {j} they \
are? Give a single integer from 0 (no change at all) to 100 (a complete change \
in how {j} they are).

Respond with ONLY a JSON object: {{"rating": <int>}}."""

_E3_CHUNK = """Below is a short excerpt of one Reddit account's posts/comments \
from a single period. Rate how strongly THIS EXCERPT expresses each trait about \
its author. Judge only from this excerpt.

Excerpt:
{chunk_text}

{json_instruction}"""

PROMPTS = Prompts(
    version="p1",
    templates={
        "evidence_brief": _EVIDENCE_BRIEF,
        "e1_elicit": _E1_ELICIT,
        "e1_ablate": _E1_ABLATE,
        "e2_pair": _E2_PAIR,
        "e2_generic": _E2_GENERIC,
        "e3_chunk": _E3_CHUNK,
    },
)
