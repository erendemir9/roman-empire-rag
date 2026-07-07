import os
import logging

# turn off Hugging Face progress bars and warnings before the import
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.disable(logging.WARNING)

from sentence_transformers import SentenceTransformer
import chromadb
import ollama

embed_model = "all-MiniLM-L6-v2"
llm_model = "qwen2.5:3b"
top_k = 5


def build_prompt(question, contexts):
    # put the retrieved chunks into the prompt as context
    context_text = "\n\n".join(contexts)
    prompt = f"""You answer questions about the Roman Empire using only the context below.
Answer in clear, simple, everyday language that is easy to understand, not in an academic or formal tone.
If the question is not about the Roman Empire, or the answer is not in the context, reply with exactly this sentence and nothing else: I can only answer questions about the Roman Empire based on the document.

Context:
{context_text}

Question: {question}"""
    return prompt


model = SentenceTransformer(embed_model)
client = chromadb.PersistentClient(path="../chroma_db")
collection = client.get_collection("rome")

print("Roman Empire Q&A - Ask me anything about the Roman Empire.")
print("Type 'exit' to quit.")

while True:
    question = input("\nQuestion (type exit to quit): ")
    if question == "exit":
        print("Goodbye! Thanks for using the Roman Empire Q&A.")
        break

    question_embedding = model.encode(question).tolist()
    results = collection.query(query_embeddings=[question_embedding], n_results=top_k)
    contexts = results["documents"][0]

    prompt = build_prompt(question, contexts)
    response = ollama.chat(model=llm_model, messages=[{"role": "user", "content": prompt}])
    print("\nAnswer:", response["message"]["content"])
