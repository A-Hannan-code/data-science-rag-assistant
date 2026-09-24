from pathlib import Path

# SentenceTransformer for embeddings...
from sentence_transformers import SentenceTransformer   

# ollama for generating answers...
# (Faceboob AI Similarity Search) faiss for saving vectors and for similarity check..
# numpy for converting vectors into array()...

import json, ollama, faiss, numpy as np

# Directory...
faiss_dir = Path("data/faiss_index")

# loading embeddings data.....
index = faiss.read_index(
    str(faiss_dir / "index.faiss")
)

# loading chunks data
with open(
    faiss_dir / "chunks.json",
    'r',
    encoding='utf-8'
) as f:
    chunks = json.load(f)

# Embedding model....
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# User question...
querry = input("Enter your question: ")

# Creating question embeddings...
querry_embedding = embedding_model.encode(
    [querry]
)

# converting embeddings into numpy array()..
# faiss uses NumPy as its memory-bridge to python..
querry_vector = np.array(
    querry_embedding,
    dtype="float32"
)

# getting distance and index for question from faiss index..
# the smaller the distance the more similarity...
distance, indices = index.search(
    querry_vector,
    k=4
)

# creating context for llm model...
context = ""

#FAISS uses NumPy arrays, so the saved data is stored in a nested format like [[]]. indices[0] extracts the first inner array, converting it into a simple array like [].
for idx in indices[0]:
    context += chunks[idx]["text"] + "\n\n"

# creating prompt for llm and providing context to answer the question..
prompt = f"""
Answer the question using the context below.

context:
{context}

Question:
{querry}

Answer:
"""

# print("\nPROMPT SENT TO LLAMA:")
# print(prompt)

# loading the llm model -> "llama3.2:3b"...
response = ollama.chat(
    model="llama3.2:3b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

# getting answer...
answer = response["message"]["content"]

print("\n Answer:")
print(answer)     # printing answer....