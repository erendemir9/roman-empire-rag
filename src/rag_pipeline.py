import re
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

pdf_path = "../data/roman_empire.pdf"
chunk_size = 500
overlap = 80


def read_pdf(path):
    # join the text of every page into one string
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text = text + page.extract_text() + " "
    return text


def clean_text(text):
    # remove line breaks and extra spaces
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def make_chunks(text):
    # slide a word window over the text with some overlap
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = start + chunk_size - overlap
    return chunks


text = read_pdf(pdf_path)
text = clean_text(text)
chunks = make_chunks(text)
print("Number of chunks:", len(chunks))

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

client = chromadb.PersistentClient(path="../chroma_db")

# start fresh so we do not add the same chunks twice
existing = [c.name for c in client.list_collections()]
if "rome" in existing:
    client.delete_collection("rome")
collection = client.create_collection("rome")

ids = []
for i in range(len(chunks)):
    ids.append("chunk_" + str(i))

collection.add(ids=ids, documents=chunks, embeddings=embeddings.tolist())

print("Database built with", len(chunks), "chunks")
