# Troubleshooting

## `streamlit: command not found`

Activate the virtual environment and install dependencies:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Sentence Transformer model cannot load

The first run may need internet access to download `sentence-transformers/all-MiniLM-L6-v2`. If the model cannot be loaded, the application uses its local TF-IDF fallback. The fallback is suitable for a classroom demonstration but is not equivalent to transformer-based semantic embeddings.

## PyTorch installation problems

Install a CPU-compatible PyTorch build using the official selector at https://pytorch.org/get-started/locally/ and then reinstall `sentence-transformers` if necessary. Avoid installing CUDA packages on machines without a compatible NVIDIA GPU.

## PDF contains no extracted text

The parser supports text-based PDFs. Scanned image-only PDFs need OCR before upload. Password-protected or malformed PDFs may not be readable.

## Upload rejected

Current safeguards:
- Maximum 5 MB per uploaded file.
- Maximum 25 PDF pages.
- Maximum 100,000 extracted characters per resume and job description.
- Maximum 50 resumes per analysis.
- DOCX archive expansion is capped to reduce decompression-bomb risk.

## Out-of-memory errors

Try fewer resumes per batch, keep files under the configured limits and use the TF-IDF fallback if the transformer model is too heavy for the deployment host.

## Run tests

```bash
python -m pytest -q
```
