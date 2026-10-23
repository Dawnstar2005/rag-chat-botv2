"""Prompt assembly for retrieval-augmented answers."""

from xml.sax.saxutils import escape, quoteattr

from .base import Message

SYSTEM_PROMPT = """You are a helpful assistant inside a document question-answering app. \
Users upload their own documents; for each question, the most relevant excerpts are retrieved \
and placed in the user's message inside <documents> tags, each with an index number.

How to use the excerpts:
- When they contain the answer, base your answer on them and cite the excerpt numbers inline \
in square brackets, e.g. [1] or [2][3]. Cite only numbers that appear in <documents>.
- If the question is about the documents but the excerpts don't contain the answer, say you \
couldn't find it in the uploaded documents instead of guessing.
- If the excerpts are unrelated to the question (small talk, general questions), ignore them \
and answer normally from your own knowledge, without citations.
- Never invent document content, quotes, numbers or page references.

Reply in the language the user writes in (usually Mongolian, in Cyrillic script). \
Use Markdown where it improves readability."""


def format_context(chunks: list[dict]) -> str:
    parts = ["<documents>"]
    for index, chunk in enumerate(chunks, start=1):
        attrs = f"index=\"{index}\" source={quoteattr(chunk['doc_name'])}"
        if chunk.get("page"):
            attrs += f" page=\"{chunk['page']}\""
        parts.append(f"<document {attrs}>\n{escape(chunk['text'])}\n</document>")
    parts.append("</documents>")
    return "\n".join(parts)


def build_messages(history: list[Message], chunks: list[dict], max_messages: int) -> list[Message]:
    """Trim and normalise the history, then attach retrieved excerpts to the last user turn.

    Excerpts are attached only to the newest question (older turns stay plain text) so
    the conversation does not grow by a full context block every turn.
    """
    trimmed = history[-max(max_messages, 1):]  # history[-0:] would keep everything
    while trimmed and trimmed[0]["role"] != "user":
        trimmed = trimmed[1:]

    merged: list[Message] = []
    for message in trimmed:
        if merged and merged[-1]["role"] == message["role"]:
            merged[-1] = {"role": message["role"], "content": merged[-1]["content"] + "\n\n" + message["content"]}
        else:
            merged.append(dict(message))

    if chunks and merged:
        question = merged[-1]["content"]
        merged[-1]["content"] = f"{format_context(chunks)}\n\n{question}"
    return merged


def retrieval_query(history: list[Message]) -> str:
    """The text used to search the documents: the latest question, plus the previous
    one when the latest is a short follow-up ("тэр хэзээ болсон бэ?")."""
    user_turns = [m["content"] for m in history if m["role"] == "user"]
    if not user_turns:
        return ""
    query = user_turns[-1]
    if len(query.split()) <= 6 and len(user_turns) > 1:
        query = f"{user_turns[-2]}\n{query}"
    return query
