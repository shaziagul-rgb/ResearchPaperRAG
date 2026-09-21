import asyncio

from app.llm import generate_grounded_summary


async def main():
    evidence = [
        {
            "page": 6,
            "text": (
                "The study uses a corpus of documents collected "
                "from several sources for empirical analysis."
            ),
        }
    ]

    result = await generate_grounded_summary(
        "Dataset",
        evidence,
    )

    print("\nLLM RESULT:")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())