import pandas as pd
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

faq_path = Path(__file__).parent / "data" / "faq_data.csv"

chroma_client = chromadb.Client()
collection_name = "faq"
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

groq_client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# ✅ Ingest FAQ data
def ingest_faq_data(faq_path):
    print("Ingesting FAQ data into ChromaDB...")
    if collection_name not in [c.name for c in chroma_client.list_collections()]:
        collection = chroma_client.create_collection(
            name=collection_name,
            embedding_function=ef
        )

        df = pd.read_csv(faq_path, encoding="latin-1")
        questions = df["Question"].to_list()
        answers = df["Answer"].to_list()
        metadata = [{"answer": ans} for ans in answers]
        ids = [f"id_{i}" for i in range(len(questions))]

        collection.add(
            documents=questions,
            metadatas=metadata,
            ids=ids
        )

        print(f"FAQ data ingested successfully into collection {collection_name}.")
    else:
        print(f"Collection {collection_name} already exists. Skipping ingestion.")


# ✅ Get top relevant FAQs from ChromaDB
def get_relevant_faq(query, n_results=2):
    collection = chroma_client.get_collection(
        name=collection_name,
        embedding_function=ef
    )
    result = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return result


# ✅ Core logic with conversation context
def faq_chain(conversation):
    """
    conversation: Full conversation history as a string, ending with the latest user query.
    """
    # Extract latest user message (last line after "User:")
    lines = conversation.strip().split("\n")
    query = lines[-1].replace("User:", "").strip()

    # Retrieve relevant FAQ context
    results = get_relevant_faq(query)
    context = " ".join([r.get('answer') for r in results["metadatas"][0]])

    # Generate contextual answer
    answer = generate_answer(conversation, context)
    return answer


def generate_answer(conversation, context):
    prompt = f"""
You are a helpful E-commerce assistant.
Use the FAQ context below to answer the user's question naturally in a conversational way.
If the user asks a follow-up like 'tell me more about that', infer what 'that' refers to
from the conversation history.

If you don't find the answer inside the context, reply with "I'm sorry, I don't know that yet."

---
Conversation so far:
{conversation}

Relevant FAQ context:
{context}

Your helpful answer:
"""

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
        )

        return chat_completion.choices[0].message.content.strip()

    except Exception as e:
        return f"An error occurred while generating the answer: {e}"


# ✅ Test script
if __name__ == "__main__":
    ingest_faq_data(faq_path)
    query = "What is your policy on defective products?"
    results = faq_chain(f"User: {query}")
    print(results)
