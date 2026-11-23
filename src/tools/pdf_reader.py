"""
src/tools/pdf_reader.py - Full-Text PDF Reader & Document Ingestion Tool
Extracts and chunks full-text research papers from local PDF files and arXiv PDF endpoints.
"""

import io
import re
import urllib.request
import logging
from typing import List, Dict, Optional, Union, Any
import pypdf

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_source: Union[str, bytes, io.BytesIO], max_pages: int = 15) -> str:
    """
    Extracts plain text from a PDF file path, raw bytes, or BytesIO object.
    Limits extraction to max_pages to preserve token budgets.
    """
    try:
        if isinstance(file_source, (bytes, bytearray)):
            stream = io.BytesIO(file_source)
        elif isinstance(file_source, io.BytesIO):
            stream = file_source
        else:
            stream = open(file_source, "rb")

        reader = pypdf.PdfReader(stream)
        pages_to_read = min(len(reader.pages), max_pages)

        extracted_text = []
        for i in range(pages_to_read):
            page_text = reader.pages[i].extract_text()
            if page_text:
                extracted_text.append(f"--- [Page {i+1}] ---\n{page_text}")

        return "\n\n".join(extracted_text)
    except Exception as e:
        logger.warning(f"Failed to extract text from PDF: {e}")
        return ""


def chunk_document_text(text: str, chunk_size: int = 1200, chunk_overlap: int = 150) -> List[str]:
    """
    Splits long document text into overlapping segments suitable for vector relevance filtering.
    """
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if len(chunk) > 100:  # Ignore trivial tiny chunks
            chunks.append(chunk)
        if end == text_len:
            break
        start += chunk_size - chunk_overlap

    return chunks


def fetch_arxiv_pdf_text(arxiv_url_or_id: str, max_pages: int = 8) -> Optional[str]:
    """
    Downloads the full-text PDF for an arXiv paper and extracts its content.
    Transforms abstract URLs like 'http://arxiv.org/abs/2401.12345' into PDF endpoints.
    """
    match = re.search(r"(\d{4}\.\d{4,5}(?:v\d+)?)", arxiv_url_or_id)
    if not match:
        return None

    arxiv_id = match.group(1)
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    try:
        req = urllib.request.Request(
            pdf_url,
            headers={"User-Agent": "AutonomousResearchAssistant/1.0 (academic research tool)"}
        )
        with urllib.request.urlopen(req, timeout=12) as response:
            pdf_bytes = response.read()

        text = extract_text_from_pdf(io.BytesIO(pdf_bytes), max_pages=max_pages)
        if text and len(text) > 500:
            return text
        return None
    except Exception as e:
        logger.warning(f"Failed to download/extract full arXiv PDF for {arxiv_id}: {e}")
        return None


def process_pdf_files(
    uploaded_files: List[Any],
    source_label: str = "Uploaded Research PDF"
) -> List[Dict]:
    """
    Processes a list of user-uploaded PDF file objects (e.g. from Streamlit file_uploader).
    Chunks each file and formats them as standard source items for RAG ranking.
    """
    sources: List[Dict] = []

    for file_obj in uploaded_files:
        filename = getattr(file_obj, "name", "Document.pdf")
        try:
            # Read bytes from file buffer
            if hasattr(file_obj, "read"):
                content_bytes = file_obj.read()
                if hasattr(file_obj, "seek"):
                    file_obj.seek(0)
            else:
                content_bytes = file_obj

            full_text = extract_text_from_pdf(io.BytesIO(content_bytes), max_pages=20)
            chunks = chunk_document_text(full_text, chunk_size=1200, chunk_overlap=150)

            for idx, ch in enumerate(chunks, 1):
                sources.append({
                    "title": f"{filename} (Section {idx})",
                    "url": f"local://{filename}#sec{idx}",
                    "published": "Local Document",
                    "content": ch,
                    "source_type": source_label,
                    "retrieved_by": "DeepDocReader"
                })
        except Exception as e:
            logger.error(f"Error processing uploaded PDF {filename}: {e}")

    return sources
