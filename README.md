# AI Research Assistant — Local RAG

A beginner-friendly Retrieval-Augmented Generation (RAG) application that lets you upload PDFs, search their content semantically, and ask questions with source-page references.

**No paid API key is required.** The language model and embedding model run locally through Ollama.

## What you need

- Windows 10/11
- Python 3.10 or 3.11 (Python 3.11 recommended)
- VS Code
- Internet connection for the initial software/model downloads
- Around 5 GB or more free disk space for Python packages and models

This project is designed to start with a small 3B model. On a 12 GB RAM laptop, close memory-heavy applications while models are running. Speed will depend on your CPU.

## 1. Install Ollama

1. Download and install Ollama for Windows: https://ollama.com/download
2. Open a **new VS Code terminal** after installation.
3. Check that it is available:

```powershell
ollama --version
```

Ollama normally runs as a background service after installation.

## 2. Download the local models

Run these commands in the VS Code terminal. The first model generates answers; the second creates embeddings for semantic search.

```powershell
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

These downloads are free, but require disk space and internet access. After downloading, inference runs locally. Model publishers' licenses still apply.

Check installed models:

```powershell
ollama list
```

## 3. Open the project in VS Code

1. Extract `PDF_RAG_Research_Assistant.zip`.
2. In VS Code, select **File → Open Folder**.
3. Select the extracted `PDF_RAG_Research_Assistant` folder.
4. Open **Terminal → New Terminal**.

## 4. Create and activate a virtual environment

In the VS Code terminal (PowerShell):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use this command for the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

If `py -3.11` is unavailable, install Python 3.11 from https://www.python.org/downloads/ and reopen VS Code.

## 5. Install Python packages

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `faiss-cpu` fails to install, check that you are using 64-bit Python 3.11. Do not install packages into a different Python environment.

## 6. Run the app

Make sure Ollama is running, then:

```powershell
python -m streamlit run app.py
```

Streamlit should open the app in your browser, usually at:

http://localhost:8501

## 7. Use the app

1. Open **Upload & index PDFs**.
2. Upload one or more text-based PDF files.
3. Keep the default chunk size (800 characters) and overlap (120 characters) for your first test.
4. Click **Process and index PDFs** and wait for the local embedding model to finish.
5. Open **Ask questions**.
6. Ask a question whose answer is present in your PDF.
7. Expand **Retrieved source passages** to inspect the chunks and page numbers used.

Try asking a question that is not answered in your PDF. The assistant is instructed to say when the context is insufficient.

## Project structure

```text
PDF_RAG_Research_Assistant/
├── app.py                 # Streamlit user interface
├── rag_pipeline.py        # PDF, chunking, embedding, FAISS, RAG logic
├── requirements.txt       # Python dependencies
├── README.md              # Setup and run instructions
├── data/
│   └── uploads/           # Uploaded PDF copies
└── vector_store/
    ├── index.faiss        # Local FAISS vector index (created at runtime)
    └── metadata.json      # Chunk text and PDF page metadata (created at runtime)
```

## How the RAG pipeline works

1. **Extract:** PyMuPDF extracts text page by page and keeps the page number.
2. **Chunk:** text is split into overlapping character windows.
3. **Embed:** Ollama `nomic-embed-text` converts each chunk into a vector.
4. **Index:** FAISS stores normalized vectors and uses inner-product search, equivalent to cosine similarity for normalized vectors.
5. **Retrieve:** the question is embedded and the top four most similar chunks are retrieved.
6. **Generate:** Ollama `llama3.2:3b` receives the question and retrieved passages and generates a grounded answer.
7. **Cite:** source filenames and page numbers are shown with retrieved passages and requested in the generated answer.

## Important limitations

- This starter version extracts selectable PDF text. Scanned/image-only PDFs need OCR, which is not included yet.
- Chunk size is measured in characters in this starter implementation, not model tokens.
- Similarity retrieval is not a guarantee that the retrieved passage contains the correct answer. Inspect the source passages.
- The app stores uploaded PDFs, the FAISS index, and metadata locally in this project folder.
- Do not upload confidential documents unless you are comfortable storing them on this computer.
- This is a learning project, not a production security-reviewed application.

## Troubleshooting

### `ollama` is not recognized
Install Ollama, close and reopen VS Code, then run `ollama --version`.

### Cannot connect to Ollama
Open the Ollama desktop app. If necessary, open another terminal and run:

```powershell
ollama serve
```

If Ollama is already running as a background service, you may not need this command.

### Model not found
Run:

```powershell
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### PowerShell says scripts are disabled
Run the temporary process-scoped execution policy command in Step 4, then activate the environment again.

### App says no text was extracted
The PDF may be scanned. Try a PDF where you can select and copy text. OCR can be added as a later enhancement.

### Out of memory or very slow
Close other applications. Use a smaller model if needed, such as `llama3.2:1b`, and change the Chat model field in the sidebar to `llama3.2:1b`.

## Stop the app

Press `Ctrl + C` in the VS Code terminal.
