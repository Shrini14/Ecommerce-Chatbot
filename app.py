import streamlit as st
from faq import ingest_faq_data, faq_chain
from pathlib import Path
from router import get_route

# Load FAQ data
faqs_path = Path(__file__).parent / "data" / "faq_data.csv"
ingest_faq_data(faqs_path)

st.title("🛒 E-commerce Chatbot")

# Initialize chat
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display previous chat
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])


def ask(query):
    # Combine entire conversation
    history = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state["messages"]]
    )

    route = get_route(query)[0]

    if route == 'faq':
        conversation = f"{history}\nUser: {query}"
        return faq_chain(conversation)
    elif route == 'sql':
        return "SQL router not yet implemented."
    else:
        return f"Route '{route}' not implemented yet."


query = st.chat_input("Ask me anything about our products or services!")

if query:
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    response = ask(query)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
