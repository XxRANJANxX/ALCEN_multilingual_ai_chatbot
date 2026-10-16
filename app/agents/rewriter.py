from app.llm.client import LLMClient, LLMError

PROMPT = (
    "Rewrite the user's last message as a standalone search query, resolving pronouns and "
    "references using the conversation. Keep the SAME language as the last message. "
    "Output only the query, nothing else."
)


async def rewrite(llm: LLMClient, message: str, history: list[dict]) -> str:
    if not history or not llm.available:
        return message
    convo = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in history[-6:])
    try:
        out = await llm.chat(
            [{"role": "system", "content": PROMPT},
             {"role": "user", "content": f"Conversation:\n{convo}\n\nLast message: {message}"}],
            temperature=0, max_tokens=120,
        )
    except LLMError:
        return message
    out = out.strip().strip('"').strip()
    return out if 0 < len(out) <= 500 else message
