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
   
   uploaded_file = st.file_uploader("Upload your document", type=["pdf", "docx", "txt","md"])
   
   if not uploaded_file:
       st.info("Please upload your document to continue", icon="📂")
   else:
       
        success_message = st.success("File uploaded successfully!", icon="✅")
        analyze_button = st.button("Analyze 🕵🏼")
        analyze_message = st.empty()
        
        if analyze_button:
            
            analyze_message.write("Analyzing...🔍")
            time.sleep(2)  
            analyze_message.empty()

            analyzed_algo = st.subheader("Encryption Algorithm : AES encryption algorithm")
            st.image("./images/aes.jpg")
                 
with tab2:
    
    cipher_text = st.text_area(
    "Ciphered Text",
    placeholder="Enter your ciphered text here :" +
    "\n4D 0F D0 D2 A0 09 F5 10 E0 8A 30 06 4D 53 A4 1F 63 4A 90 29 4D 0F D0 D2 A0 09 F5 10 E0 8A 30 06 4D 53 A4 1F 63 4A 90 29",
    height=150,  # Adjust height as needed
    )
    
    info_placeholder = st.empty()
    
    if not cipher_text.strip():  
        info_placeholder.info("Please add your cipher text to continue.", icon="🗝️")
    else:
        info_placeholder.empty()
    
        analyze_button = st.button("Analyze 🕵🏼")
        analyze_message = st.empty()
            
        if analyze_button:   
            analyze_message.write("Analyzing...🔍")
            time.sleep(2)  

            analyze_message.empty()

            analyzed_algo = st.subheader("Encryption Algorithm : AES encryption algorithm")
            st.image("./images/aes.jpg")
    
    
    
    
    
