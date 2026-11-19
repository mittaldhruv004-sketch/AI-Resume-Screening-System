"""Safe text extraction for PDF, DOCX and TXT resumes."""
from io import BytesIO
from zipfile import BadZipFile, ZipFile
import io
import re
import pdfplumber
from pypdf import PdfReader
from docx import Document

MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_PDF_PAGES = 25
MAX_TEXT_CHARS = 100_000


def extract_text_from_file(filename: str, buffer: BytesIO) -> str:
    data = buffer.getvalue()
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("File exceeds the 5 MB upload limit.")
    lower = filename.lower()
    if lower.endswith(".pdf"):
        if not data.startswith(b"%PDF-"):
            raise ValueError("The uploaded file is not a valid PDF.")
        text_parts = []
        try:
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                if len(pdf.pages) > MAX_PDF_PAGES:
                    raise ValueError("PDF exceeds the 25-page limit.")
                for page in pdf.pages:
                    text_parts.append(page.extract_text() or "")
        except ValueError:
            raise
        except Exception:
            reader = PdfReader(io.BytesIO(data), strict=False)
            if len(reader.pages) > MAX_PDF_PAGES:
                raise ValueError("PDF exceeds the 25-page limit.")
            text_parts = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(text_parts)
    elif lower.endswith(".docx"):
        if not data.startswith(b"PK"):
            raise ValueError("The uploaded file is not a valid DOCX document.")
        try:
            with ZipFile(io.BytesIO(data)) as archive:
                members = archive.infolist()
                if len(members) > 2000 or sum(member.file_size for member in members) > 25 * 1024 * 1024:
                    raise ValueError("DOCX expands beyond the safe document limit.")
            doc = Document(io.BytesIO(data))
            parts = [p.text for p in doc.paragraphs]
            for table in doc.tables:
                for row in table.rows:
                    parts.append(" | ".join(cell.text for cell in row.cells))
            text = "\n".join(parts)
        except (BadZipFile, Exception) as exc:
            raise ValueError("Unable to read this DOCX file.") from exc
    elif lower.endswith(".txt"):
        text = data.decode("utf-8-sig", errors="replace")
    else:
        raise ValueError("Unsupported file type. Use PDF, DOCX, or TXT.")
    text = re.sub(r"\x00", "", text)
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError("Extracted text exceeds the 100,000-character limit.")
    if not text.strip():
        raise ValueError("No readable text found. This may be a scanned PDF; use OCR before uploading.")
    return text
