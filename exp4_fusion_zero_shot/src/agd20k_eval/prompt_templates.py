"""
Prompt templates for 25 affordance categories in AGD20K Unseen split.

Each entry maps the affordance name to:
    - `prompt`: f-string style template accepting `object`.
    - `tokens`: list of token substrings expected to align with the action/part.

These tokens are used to pick the correct heatmap (matching tokenized output from tokenizer_2).
"""

PROMPT_TEMPLATES = {
    "carry": {
        "prompt": "carry {object}",
        "tokens": ["▁carry"],
    },
    "catch": {
        "prompt": "catch {object}",
        "tokens": ["▁catch"],
    },
    "cut": {
        "prompt": "cut {object}",
        "tokens": ["▁cut"],
    },
    "cut_with": {
        "prompt": "{object} cut",
        "tokens": ["▁cut"],
    },
    "drink_with": {
        "prompt": "{object} drink",
        "tokens": ["▁drink"],
    },
    "eat": {
        "prompt": "{object} eat",
        "tokens": ["▁eat", "eat"],
    },
    "hit": {
        "prompt": "hit {object}",
        "tokens": ["▁hit"],
    },
    "hold": {
        "prompt": "hold {object}",
        "tokens": ["▁hold"],
    },
    "jump": {
        "prompt": "jump {object}",
        "tokens": ["▁jump"],
    },
    "kick": {
        "prompt": "kick {object}",
        "tokens": ["▁kick"],
    },
    "lie_on": {
        "prompt": "lie on {object}",
        "tokens": ["▁lie"],
    },
    "open": {
        "prompt": "open {object}",
        "tokens": ["▁open"],
    },
    "peel": {
        "prompt": "peel {object}",
        "tokens": ["▁peel"],
    },
    "pick_up": {
        "prompt": "pick up {object}",
        "tokens": ["▁pick", "up"],
    },
    "pour": {
        "prompt": "pour {object}",
        "tokens": ["▁pour"],
    },
    "push": {
        "prompt": "push {object}",
        "tokens": ["▁push"],
    },
    "ride": {
        "prompt": "ride {object}",
        "tokens": ["▁ride"],
    },
    "sip": {
        "prompt": "sip {object}",
        "tokens": ["▁sip","_si"],
    },
    "sit_on": {
        "prompt": "sit on {object}",
        "tokens": ["▁sit"],
    },
    "stick": {
        "prompt": "{object} stick",
        "tokens": ["▁stick"],
    },
    "swing": {
        "prompt": "swing {object}",
        "tokens": ["▁swing"],
    },
    "take_photo": {
        "prompt": "take a photo with {object}",
        "tokens": ["▁take", "photo"],
    },
    "throw": {
        "prompt": "throw {object}",
        "tokens": ["▁throw"],
    },
    "type_on": {
        "prompt": "type on the {object}",
        "tokens": ["▁type"],
    },
    "wash": {
        "prompt": "wash {object}",
        "tokens": ["▁wash"],
    },
}
