# for embeddings..
from sentence_transformers import SentenceTransformer
from pathlib import Path

# for accessing .env file and API key..
from dotenv import load_dotenv
from google import genai


# Streamlit for app...
# numpy because faiss expects numpy array()...
# faiss for accessing the saved embeddings and chunks data with similarity..
import json, faiss, numpy as np, os, streamlit as st


@st.cache_resource # Cache the resource so it isn't loaded again on every app rerun.
def load_rag_data():

    faiss_dir = Path("data/faiss_index")

    index = faiss.read_index(
        str(faiss_dir / "index.faiss")
    )

    with open(faiss_dir / "chunks.json", 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    return index, chunks

index, chunks = load_rag_data()


load_dotenv()  # load the .env (API KEY FILE)...

# Create a Gemini API client using the API key stored in the environment variable.
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


# Cache the resource so the embedding model isn't loaded again on every app rerun.
@st.cache_resource 
def load_embedding_model():  

    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
embedding_model = load_embedding_model()   # loading the model..


def ask_rag(question_query):

    # Create a Gemini chat client for splitting the user's message into questions.
    chat = client.chats.create(
        model="gemini-3.5-flash-lite"
        )

    # Prompt Gemini to split a message containing multiple questions into separate questions.
    split_prompt = f"""
    Split the user's message into separate questions.

    Rules:
    - Return only the questions.
    - Put each question on a new line.
    - Do not answer them.
    - Do not add explanations.
    - If the message contains only one question, return that question unchanged.

    User message:
    {question_query}
    """

    # Send the user's message to Gemini and get the separated questions.
    split_response = chat.send_message(split_prompt)

    # Convert Gemini's response into a clean list of questions.
    questions = [
        q.strip()
        for q in split_response.text.splitlines()
        if q.strip()
    ]


    # Store the retrieved context and source information.
    context = ""
    sources = {}

    # Set the maximum FAISS distance allowed for a chunk to be considered relevant..
    threshold = 1.35

    # Process each question separately...
    for question in questions:

        # convert the question into an embedding vector....
        query_embedding = embedding_model.encode(
            [question]
        )

        # convert the vector into numpy array required by faiss...
        query_vector = np.array(
            query_embedding,
            dtype="float32"
        )

        # search FAISS for the 10 closest chunks to the question...
        distances, indices = index.search(
            query_vector,
            k=10
        )


        # search each retrieved chunk...
        for i, idx in enumerate(indices[0]):

            distance = distances[0][i]

            # ignore chunks whose distance is > threshold...
            if distance > threshold:
                continue

            # get the chunk using faiss index...
            chunk = chunks[idx]

            # add chunk text to context...
            context += chunk["text"] + "\n\n"

            # extract metadata from chunk..
            document = chunk["metadata"]["document"]
            page = chunk["metadata"]["page"]


            # store the document and page number in sources {}...
            if document not in sources:
                sources[document] = page  

        # if no chunk is found return this response...
        if not context.strip():

            return (
                "I couldn't find relevant information in the knowledge base.",
                {}
            )


    # creating prompt for gemini...
    prompt = f"""
    Use the following context to answer the user's questions.

    Rules:

    1. Answer each question separately. Use only the information provided in the context. Do not use outside knowledge.

    2. Explain the answer in simple, clear, and beginner-friendly language.

    3. If the user asks for an example, give an example specifically related to that question. Do not include irrelevant examples or information.

    4. If an example in the context is complicated, simplify it while keeping its original meaning.

    5. If the answer in the context is complicated or difficult to understand, explain it in simpler language without changing its meaning.

    6. If the context contains a short or incomplete explanation, you may add 1-2 simple sentences to make it easier to understand, but only using information supported by the context.

    7. Do not add information from your own knowledge.

    8. Do not include raw prompt/template text, such as \\n, {{text}}, or similar formatting placeholders, in the answer.

    9. Keep the answer focused on each question and do not add unrelated information.

    10. If a question contains multiple parts and the context supports only some of them, answer the parts supported by the context and clearly state which part cannot be answered from the knowledge base.

    11. Do not refuse an entire question just because one part of it is missing from the context. Answer the part that is supported by the context.

    12. If you cannot find any answer for a particular question do not give the sources then.

    Context:
    {context}

    Questions:
    {questions}

    Answer:
    """


    # Create a Gemini chat client
    chat = client.chats.create(
        model="gemini-3.5-flash-lite"
    )

    # send the prompt with retrieved context to gemini for answer..
    response = chat.send_message(prompt)

    return response.text, sources