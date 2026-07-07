# Roman Empire RAG

A simple RAG (retrieval-augmented generation) question-answering system about the Roman Empire.

## Files

- `src/generate_pdf.py` - turns the text file into a PDF.
- `src/rag_pipeline.py` - reads the PDF, splits it into chunks, makes embeddings, and stores them in the vector database.
- `src/query.py` - ask questions and get answers from the local LLM.
- `src/evaluate.py` - tests the system on some questions and saves the results.
- `src/app.py` - web interface, same Q&A as query.py.

## Installation

Create and activate a virtual environment, then install the dependencies:

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the virtual environment with:

```
venv\Scripts\activate
```

## How to run

Run these from inside the `src` folder, in this order:

```
python generate_pdf.py
python rag_pipeline.py
python query.py
python evaluate.py   (optional, creates the results files)
streamlit run app.py   (optional web interface, run from the src folder)
```

## Note

You need Ollama running with the `qwen2.5:3b` model.
