import streamlit as st
from openai import OpenAI
import time



with st.sidebar:
    st.header("Ciphery")
    st.write("A simple Streamlit app that analyses the algorithm used in a given ciphered database.")
    st.info("The application is currently under development and will be released soon")
        
        
# Show title and description.
st.title("📄 Ciphery")
st.write(
    "A simple Streamlit app that analyses the algorithm used in a given ciphered database."
)   

tab1,tab2, = st.tabs(["📂 Upload a file", "⚡ Enter text manually"])

with tab1:
   
   uploaded_file = st.file_uploader("Upload your document", type=["pdf", "docx", "txt"])
   
   if not uploaded_file:
       st.info("Please upload your document to continue", icon="📂")
   else:
        success_message = st.success("File uploaded successfully!", icon="✅")
        time.sleep(3)
        success_message.empty()
           
with tab2:
    
    cipher_text = st.text_area(
    "Ciphered Text",
    placeholder="Enter your ciphered text here. Example:\
        ",
    height=150,  # Adjust height as needed
    )
    
    st.info("Please add your cipher text to continue.", icon="🗝️")
    
    
    
