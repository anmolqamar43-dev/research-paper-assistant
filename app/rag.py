from pathlib import Path
import time
import pdfplumber
from pypdf import PdfReader


def extract_pdf_content(file_path: str):
    """
    Extract text from a PDF.

    PyPDF is tried first.
    PDFPlumber is used as a fallback.

    Each extracted page keeps:
    - document name
    - page number
    - page text
    """

    pdf_path = Path(file_path)
    document_name = pdf_path.name

    pages = []

    # ---------------------------------------------
    # Try PyPDF
    # ---------------------------------------------

    try:
        reader = PdfReader(str(pdf_path))

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()

            if text:
                pages.append(
                    {
                        "document": document_name,
                        "page": page_number,
                        "text": text,
                    }
                )

    except Exception:
        pages = []


    # ---------------------------------------------
    # Use PDFPlumber if PyPDF found nothing
    # ---------------------------------------------

    if not pages:
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:

                for page_number, page in enumerate(
                    pdf.pages,
                    start=1
                ):
                    text = page.extract_text() or ""
                    text = text.strip()

                    if text:
                        pages.append(
                            {
                                "document": document_name,
                                "page": page_number,
                                "text": text,
                            }
                        )

        except Exception as error:
            raise RuntimeError(
                f"Unable to read PDF: {error}"
            )


    return pages


# ---------------------------------------------
# Create Text Chunks
# ---------------------------------------------

def create_chunks(
    pages,
    chunk_size=1000,
    overlap=150
):
    """
    Split page text into smaller chunks.

    Document name and page number are preserved.
    """

    chunks = []

    for page in pages:

        text = page["text"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end].strip()

            if chunk_text:

                chunks.append(
                    {
                        "document": page["document"],
                        "page": page["page"],
                        "text": chunk_text,
                    }
                )

            start += chunk_size - overlap


    return chunks


def build_context(results):
    """
    Combine retrieved chunks into a context
    that will be given to the LLM.
    """

    context_parts = []

    for result in results:

        context_parts.append(
            f"Document: {result['document']}\n"
            f"Page: {result['page']}\n"
            f"Content:\n{result['text']}"
        )

    return "\n\n---\n\n".join(context_parts)


def answer_question(client, question, results):
    """
    Generate an answer using only the retrieved
    research-paper context.
    """

    context = build_context(results)

    if not context:
        return {
            "answer": "This information is not available in the uploaded documents.",
            "sources": []
        }

    prompt = f"""
You are a Research Paper Assistant.

Answer the user's question ONLY using the information
provided in the research-paper context below.

If the answer is not present in the context, say exactly:

"This information is not available in the uploaded documents."

Do not use outside knowledge.
Do not make up information.

Research-paper context:
-----------------------
{context}
-----------------------

User question:
{question}

Give a clear and concise answer.
"""

    response = None
    last_error = None

    models = [
        "gemini-3.8-flash",
        "gemini-3.6-flash"
    ]

    for model_name in models:

        for attempt in range(3):

            try:

                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )

                break

            except Exception as error:

                last_error = error

                if "503" in str(error) or "UNAVAILABLE" in str(error):

                    if attempt < 2:
                        import time
                        time.sleep(2 ** attempt)
                        continue

                break

        if response is not None:
            break

    if response is None:
        raise RuntimeError(
            f"Gemini model temporarily unavailable. "
            f"Please try again shortly. Details: {last_error}"
        )

    sources = []

    for result in results:

        source = {
            "document": result["document"],
            "page": result["page"]
        }

        if source not in sources:
            sources.append(source)

    return {
        "answer": response.text,
        "sources": sources
    }