def extract_text(content) -> str:
    """Flatten a LangChain message's content to plain text.

    Providers disagree on shape: Groq returns a plain string, Gemini returns a
    list of content blocks, e.g. [{"type": "text", "text": "hi", ...}].
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )

    return ""
