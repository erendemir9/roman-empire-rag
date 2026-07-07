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
import streamlit as st

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


@st.cache_resource
def load_model_and_collection():
    # load once and keep it in memory between clicks
    model = SentenceTransformer(embed_model)
    client = chromadb.PersistentClient(path="../chroma_db")
    collection = client.get_collection("rome")
    return model, collection


model, collection = load_model_and_collection()

st.title("Roman Empire Q&A")
st.caption("Ask me anything about the Roman Empire.")

question = st.text_input("Your question")

if question:
    question_embedding = model.encode(question).tolist()
    results = collection.query(query_embeddings=[question_embedding], n_results=top_k)
    contexts = results["documents"][0]
    retrieved_ids = results["ids"][0]

    prompt = build_prompt(question, contexts)
    response = ollama.chat(model=llm_model, messages=[{"role": "user", "content": prompt}])
    answer = response["message"]["content"]
    st.write(answer)

    # only show the sources when we got a real answer, not the refusal
    if "I can only answer questions about the Roman Empire" not in answer:
        with st.expander("Sources"):
            # show each retrieved chunk id and its text
            for i in range(len(retrieved_ids)):
                st.markdown("**" + retrieved_ids[i] + "**")
                st.caption(contexts[i])
                st.divider()
