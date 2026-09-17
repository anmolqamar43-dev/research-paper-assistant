from pathlib import Path
import os
import shutil

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from dotenv import load_dotenv

from app.rag import (
    extract_pdf_content,
    create_chunks,
    answer_question
)

from app.vector_store import (
    create_gemini_client,
    build_vector_store,
    search_documents
)


# --------------------------------------------------
# Load Environment Variables
# --------------------------------------------------

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is missing from .env"
    )

gemini_client = create_gemini_client(
    GEMINI_API_KEY
)


# --------------------------------------------------
# FastAPI Application
# --------------------------------------------------

app = FastAPI(
    title="Research Paper Assistant",
    version="1.0.0"
)


# --------------------------------------------------
# Project Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_DIR = BASE_DIR / "uploads"
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

UPLOAD_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Templates & Static Files
# --------------------------------------------------

templates = Jinja2Templates(
    directory=str(TEMPLATE_DIR)
)

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static"
)


# --------------------------------------------------
# Home Page
# --------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "message": None,
            "papers": [],
            "errors": [],
            "answer": None,
            "sources": []
        }
    )


# --------------------------------------------------
# Upload Research Papers
# --------------------------------------------------

@app.post("/upload", response_class=HTMLResponse)
async def upload_papers(
    request: Request,
    files: list[UploadFile] = File(...)
):

    papers = []
    errors = []

    # All chunks from all uploaded PDFs
    all_chunks = []


    # --------------------------------------------------
    # Process Each PDF
    # --------------------------------------------------

    for file in files:

        # ------------------------------------------
        # Empty filename
        # ------------------------------------------

        if not file.filename:
            continue


        # ------------------------------------------
        # PDF validation
        # ------------------------------------------

        if not file.filename.lower().endswith(".pdf"):

            errors.append(
                f"{file.filename}: Only PDF files are allowed."
            )

            continue


        # ------------------------------------------
        # Save PDF
        # ------------------------------------------

        filename = Path(file.filename).name

        file_path = UPLOAD_DIR / filename

        with file_path.open("wb") as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )


        # ------------------------------------------
        # Extract PDF content
        # ------------------------------------------

        try:

            pages = extract_pdf_content(
                str(file_path)
            )

            if not pages:

                errors.append(
                    f"{filename}: No readable text found."
                )

                file_path.unlink(
                    missing_ok=True
                )

                continue


            # --------------------------------------
            # Create text chunks
            # --------------------------------------

            chunks = create_chunks(
                pages
            )


            # --------------------------------------
            # Add chunks to combined list
            # --------------------------------------

            all_chunks.extend(chunks)


            # --------------------------------------
            # Store paper information
            # --------------------------------------

            papers.append(
                {
                    "filename": filename,
                    "pages": len(pages),
                    "chunks": len(chunks)
                }
            )


        except Exception as error:

            errors.append(
                f"{filename}: {str(error)}"
            )

            file_path.unlink(
                missing_ok=True
            )


    # --------------------------------------------------
    # Build ONE FAISS Vector Store
    # --------------------------------------------------

    if all_chunks:

        try:

            build_vector_store(
                gemini_client,
                all_chunks
            )

        except Exception as error:

            errors.append(
                f"Vector database error: {str(error)}"
            )

            papers = []


    # --------------------------------------------------
    # Response Message
    # --------------------------------------------------

    if papers:

        message = (
            f"{len(papers)} research paper(s) "
            "uploaded and processed successfully."
        )

    else:

        message = (
            "No valid research papers were uploaded."
        )


    # --------------------------------------------------
    # Return HTML Page
    # --------------------------------------------------

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "message": message,
            "papers": papers,
            "errors": errors,
            "answer": None,
            "sources": []
        }
    )


# --------------------------------------------------
# Ask Question
# --------------------------------------------------

@app.post("/ask", response_class=HTMLResponse)
async def ask_question(
    request: Request
):

    # ----------------------------------------------
    # Read form data
    # ----------------------------------------------

    form = await request.form()

    question = form.get(
        "question",
        ""
    )


    # Make sure question is a string
    if not isinstance(question, str):

        question = ""


    question = question.strip()


    # ----------------------------------------------
    # Empty question
    # ----------------------------------------------

    if not question:

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "message": "Please enter a question.",
                "papers": [],
                "errors": [],
                "answer": None,
                "sources": []
            }
        )


    # ----------------------------------------------
    # Retrieve relevant documents
    # ----------------------------------------------

    try:

        results = search_documents(
            gemini_client,
            question,
            top_k=5
        )


        # ------------------------------------------
        # Generate answer
        # ------------------------------------------

        result = answer_question(
            gemini_client,
            question,
            results
        )


        # ------------------------------------------
        # Return answer
        # ------------------------------------------

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "message": None,
                "papers": [],
                "errors": [],
                "answer": result["answer"],
                "sources": result["sources"]
            }
        )


    # ----------------------------------------------
    # No vector database
    # ----------------------------------------------

    except FileNotFoundError as error:

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "message": str(error),
                "papers": [],
                "errors": [],
                "answer": None,
                "sources": []
            }
        )


    # ----------------------------------------------
    # Other errors
    # ----------------------------------------------

    except Exception as error:

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "message": None,
                "papers": [],
                "errors": [str(error)],
                "answer": None,
                "sources": []
            }
        )