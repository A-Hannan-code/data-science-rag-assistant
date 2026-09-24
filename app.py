import streamlit as st
from rag import ask_rag

# configure the streamlit page..
st.set_page_config(
    page_title="Data Science Assistant",
    page_icon="🤖",
    layout="centered"
)

# app title and description...
st.title("🤖 Data Science Assistant")

st.caption(
    "Ask questions about your Data Science course material."
)

# creating a clear chat history button..
if st.button("🧈 Clear Chat"):

    st.session_state.messages = []

    st.rerun() 

# Initialize chat history if it doesn't already exist.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display all previous messages from the chat history.
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])

        if message["role"] == "assistant":

            sources = message.get("sources", {})

            if sources:

                with st.expander("**📚 Sources**"):

                    for document, page in sources.items():

                        st.write(
                            f"- {document} --- Page: {page}"
                        )


# create chat input box..
question = st.chat_input(
    "Ask a question..."
)

if question:

    # display the user question...
    with st.chat_message("user"):
        st.write(question)

    # save the question to chat history...
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # Send the question to the RAG system and get the answer and sources.
    answer, sources = ask_rag(question)


    # Display the generated answer and its sources.
    with st.chat_message("assistant"):
        st.write(answer)

        if sources:

            with st.expander("**📚 Sources**"):

                for document, page in sources.items():

                    st.write(
                        f"- {document} --- Page: {page}"
                    )

    # save the assistant's answer and sources to chat history...
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })