import streamlit as st
from openai import OpenAI

# Show title and description.
st.title("📄 Ciphery")
st.write(
    "A simple Streamlit app that analyses the algorithm used in a given ciphered database."
)


cipher_text= st.text_input("Ciphered Text", type="password")

st.info("Please add your cipher text to continue.", icon="🗝️")


    # Let the user upload a file via `st.file_uploader`.
uploaded_file = st.file_uploader(
        "Upload a document (.txt or .md)", type=("txt", "md")
    )

