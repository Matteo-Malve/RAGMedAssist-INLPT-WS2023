import streamlit as st
from streamlit_chat import message
import yaml

# Streamlit app
st.title("Pubmed chat")

from custom_chatbot import MedicalChatbot

with open("cfg-matteo.yaml", "r") as file:
        cfg = yaml.safe_load(file)

chatbot = MedicalChatbot(cfg)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"],unsafe_allow_html=True)

# React to user input
if question := st.chat_input("Ask a medical question:"):
    # Display user message in chat message container
    st.chat_message("user").markdown(question)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})
    markdown_response=chatbot.generate_response_by_type(question)
    with st.chat_message("assistant"):
        st.markdown(markdown_response, unsafe_allow_html=True)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", 
                                          "content": markdown_response})
