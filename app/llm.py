import httpx

from .config import OLLAMA_MODEL, OLLAMA_URL


async def generate_grounded_summary(
    category: str,
    evidence: list[dict],
) -> str:
    """
    Generate a short explanation using only evidence
    retrieved from the research paper.
    """

    if not evidence:
        return "Insufficient evidence."

    evidence_text = "\n\n".join(
        f"Page {item['page']}:\n{item['text']}"
        for item in evidence
    )

    prompt = f"""You are a research paper analysis tool.

Category:
{category}

Evidence from the paper:
{evidence_text}

Write exactly two short sentences explaining what the evidence
says about this category.

Use ONLY the supplied evidence.
Do not add outside information.
Do not guess.
Do not explain your reasoning.
Return ONLY the final answer.
"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 60,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                OLLAMA_URL,
                json=payload,
            )

        response.raise_for_status()

        data = response.json()

        result = data.get("response", "").strip()

        if result:
            return result

        return "Insufficient evidence."

    except Exception as exc:
        print(f"LLM error: {exc}")
        return "LLM explanation unavailable."