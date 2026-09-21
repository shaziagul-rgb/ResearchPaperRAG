import fitz


def extract_pdf_pages(pdf_bytes: bytes) -> list[dict]:
    """
    Extract selectable text from a PDF while keeping
    the original page number.
    """

    pages: list[dict] = []

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf",
    )

    try:
        for page_number, page in enumerate(
            document,
            start=1,
        ):
            text = page.get_text("text")

            # PDFs often contain many unnecessary line breaks.
            text = " ".join(text.split())

            pages.append(
                {
                    "page": page_number,
                    "text": text,
                }
            )

    finally:
        document.close()

    return pages