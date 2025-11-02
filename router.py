import cohere
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv

# Load the API key from your custom env file
load_dotenv("router.env")
co = cohere.Client(os.getenv("COHERE_API_KEY"))

# ---- Define your routes ----
routes = {
    "faq": [
        "What is the return policy of the products?",
        "Do I get discount with the HDFC credit card?",
        "How can I track my order?",
        "What payment methods are accepted?",
        "How long does it take to process a refund?",
    ],
    "sql": [
        "I want to buy nike shoes that have 50% discount.",
        "Are there any shoes under Rs. 3000?",
        "Do you have formal shoes in size 9?",
        "Are there any Puma shoes on sale?",
        "What is the price of puma running shoes?",
    ],
}

# ---- Get embeddings ----
def get_embeddings(texts, input_type="search_document"):
    response = co.embed(
        texts=texts,
        model="embed-english-v3.0",
        input_type=input_type  # ✅ REQUIRED argument
    )
    return np.array(response.embeddings)

# ---- Precompute route embeddings ----
route_embeddings = {
    route: get_embeddings(utterances, input_type="search_document")
    for route, utterances in routes.items()
}

# ---- Router ----
def get_route(user_query: str, threshold: float = 0.25):
    query_emb = get_embeddings([user_query], input_type="search_query")
    best_route, best_score = None, -1

    for route, embs in route_embeddings.items():
        score = np.max(cosine_similarity(query_emb, embs))
        if score > best_score:
            best_route, best_score = route, score

    return best_route if best_score >= threshold else "unknown", float(best_score)

# ---- Test ----
if __name__ == "__main__":
    print(get_route("i got a defective order.what should i do?"))
    print(get_route("Do you offer international shipping?"))
