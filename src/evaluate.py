import os
import time
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

questions = [
    "What was the basic unit of the Roman army?",
    "Who was the first Roman emperor?",
    "How much were Roman legionaries paid per year?",
    "What was the Roman Republic's written law code called?",
    "What happened at the Battle of the Teutoburg Forest?",
    "What was the Colosseum used for?",
    "Which emperor made Christianity tolerated in the Roman Empire?",
    "What caused the fall of the Western Roman Empire?",
]


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


def ask_llm(prompt):
    # send one message to the local model and return the text answer
    response = ollama.chat(model=llm_model, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]


model = SentenceTransformer(embed_model)
client = chromadb.PersistentClient(path="../chroma_db")
collection = client.get_collection("rome")

qa_blocks = []
beforeafter_blocks = []
times = []

for i in range(len(questions)):
    question = questions[i]

    # before: plain question with no context
    before_answer = ask_llm(question)

    # after: retrieve chunks and answer from context
    question_embedding = model.encode(question).tolist()
    results = collection.query(query_embeddings=[question_embedding], n_results=top_k)
    contexts = results["documents"][0]
    retrieved_ids = results["ids"][0]

    prompt = build_prompt(question, contexts)
    start = time.time()
    after_answer = ask_llm(prompt)
    end = time.time()
    response_time = end - start
    times.append(response_time)

    qa_block = "Question: " + question + "\n"
    qa_block = qa_block + "Answer: " + after_answer + "\n"
    qa_block = qa_block + "Response time: " + str(round(response_time, 2)) + " s\n"
    qa_block = qa_block + "Retrieved chunks: " + str(retrieved_ids) + "\n"
    qa_block = qa_block + "----------------------------------------\n"
    qa_blocks.append(qa_block)

    ba_block = "Question: " + question + "\n"
    ba_block = ba_block + "BEFORE (no RAG): " + before_answer + "\n"
    ba_block = ba_block + "AFTER (with RAG): " + after_answer + "\n"
    ba_block = ba_block + "----------------------------------------\n"
    beforeafter_blocks.append(ba_block)

    print("Done", str(i + 1) + "/" + str(len(questions)))

if not os.path.exists("../results"):
    os.makedirs("../results")

average_time = sum(times) / len(times)

qa_file = open("../results/results_qa.txt", "w", encoding="utf-8")
for block in qa_blocks:
    qa_file.write(block)
qa_file.write("Average response time: " + str(round(average_time, 2)) + " s\n")
qa_file.close()

ba_file = open("../results/results_beforeafter.txt", "w", encoding="utf-8")
for block in beforeafter_blocks:
    ba_file.write(block)
ba_file.close()

print("Average response time:", round(average_time, 2), "s")
print("Done, results written to ../results/")
