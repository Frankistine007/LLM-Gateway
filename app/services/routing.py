# Rough heuristic guess at prompt difficulty, same spirit as estimate_tokens()
# in rate_limit.py: cheap and approximate rather than exact, because getting an
# exact answer (e.g. asking an LLM to judge the prompt) would cost more than
# the routing decision saves. No training data exists yet to justify anything
# fancier — see routing tiers below for where this plugs in.

COMPLEX_PROMPT_CHAR_THRESHOLD = 800

# Presence of any of these is treated as "this benefits from a bigger model
# even if short" — a 40-character prompt asking for a SQL query is not the
# same difficulty as a 40-character prompt asking for a joke.
CODE_MARKERS = (
    "```",
    "def ",
    "class ",
    "import ",
    "function ",
    "SELECT ",
    "for (",
    "for(",
    "while (",
    "while(",
    "public static",
    "#include",
)


def classify_prompt(messages: list[dict]) -> str:
    """Returns "simple" or "complex" for a chat message list.

    Two signals only: total length, and whether the prompt looks like a code
    task. Neither is proven against real traffic yet — this is a starting
    point to log against and revisit, not a tuned model.
    """
    text = " ".join(m.get("content", "") or "" for m in messages)

    if len(text) > COMPLEX_PROMPT_CHAR_THRESHOLD:
        return "complex"

    if any(marker in text for marker in CODE_MARKERS):
        return "complex"

    return "simple"
