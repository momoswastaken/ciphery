import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import numpy as np
import zlib
from itertools import groupby
from sklearn.feature_extraction.text import CountVectorizer
import time

# Feature extraction functions (same as before)
def calculate_frequency_within_blocks(ciphertext, block_size=8):
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(block_size, block_size))
    vectorizer.fit([ciphertext])
    return sum(vectorizer.transform([ciphertext]).toarray()[0])

def calculate_runs_index(ciphertext):
    runs = [len(list(group)) for _, group in groupby(ciphertext)]
    return len(runs)

def calculate_serial_index(ciphertext):
    n = len(ciphertext)
    return sum(ciphertext[i] == ciphertext[i+1] for i in range(n-1)) / (n-1)

def calculate_bit_entropy(ciphertext):
    return -sum((p := ciphertext.count(bit) / len(ciphertext)) * np.log2(p) for bit in '01')

def calculate_chi_square(ciphertext):
    freq = np.array([ciphertext.count(bit) for bit in '01'])
    expected = np.mean(freq)
    return np.sum(((freq - expected) ** 2) / expected)

def calculate_key_length_guess(ciphertext):
    return len(ciphertext) // 8

def calculate_autocorrelation(ciphertext):
    n = len(ciphertext)
    return sum(ciphertext[i] == ciphertext[(i+1) % n] for i in range(n)) / n

def calculate_n_gram_analysis(ciphertext, n=2):
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n))
    vectorizer.fit([ciphertext])
    return sum(vectorizer.transform([ciphertext]).toarray()[0])

def calculate_compression_ratio(ciphertext):
    compressed = zlib.compress(ciphertext.encode())
    return len(compressed) / len(ciphertext)

def calculate_hamming_weight(ciphertext):
    return ciphertext.count('1')

def calculate_average_bit_flips(ciphertext1, ciphertext2):
    return sum(bit1 != bit2 for bit1, bit2 in zip(ciphertext1, ciphertext2)) / len(ciphertext1)

def calculate_all_features(ciphertext):
    features = {
        'frequency_within_blocks': calculate_frequency_within_blocks(ciphertext),
        'runs_index': calculate_runs_index(ciphertext),
        'serial_index': calculate_serial_index(ciphertext),
        'bit_entropy': calculate_bit_entropy(ciphertext),
        'chi_square': calculate_chi_square(ciphertext),
        'key_length_guess': calculate_key_length_guess(ciphertext),
        'autocorrelation': calculate_autocorrelation(ciphertext),
        'n_gram_analysis': calculate_n_gram_analysis(ciphertext),
        'compression_ratio': calculate_compression_ratio(ciphertext),
        'hamming_weight': calculate_hamming_weight(ciphertext),
        'average_bit_flips': calculate_average_bit_flips(ciphertext, ciphertext)  # Simulated
    }
    return features

def classify_new_ciphertext(ciphertext, selector, scaler, svm_model, nn_model, label_encoder):
    new_features = calculate_all_features(ciphertext)
    new_features_df = pd.DataFrame([new_features])
    new_features_selected = selector.transform(new_features_df)
    new_features_scaled = scaler.transform(new_features_selected)

    predicted_algo_svm = svm_model.predict(new_features_scaled)
    predicted_algo_nn = nn_model.predict(new_features_scaled)

    probabilities_svm = svm_model.predict_proba(new_features_scaled)[0]

    predicted_algo_svm = label_encoder.inverse_transform(predicted_algo_svm)
    predicted_algo_nn = label_encoder.inverse_transform(predicted_algo_nn)

    return predicted_algo_svm[0], predicted_algo_nn[0], probabilities_svm

# Load dataset and train models
def train_models():
    data = pd.read_csv('crypto dataset almost final - Copy.csv')
    X = data.drop(columns=['algorithm'])
    y = data['algorithm']

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    selector = SelectKBest(f_classif, k=10)
    X_new = selector.fit_transform(X.drop(columns=['ciphertext']), y)

    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    svm_model = SVC(probability=True, random_state=42)
    svm_model.fit(X_train, y_train)

    nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    nn_model.fit(X_train, y_train)

    return selector, scaler, svm_model, nn_model, label_encoder

selector, scaler, svm_model, nn_model, label_encoder = train_models()

# Streamlit app layout
with st.sidebar:
    st.header("Ciphery")
    st.write("A simple Streamlit app that analyses the algorithm used in a given ciphered database.")
    st.info("The application is currently under development and will be released soon")

st.title("📄 Ciphery")
st.write("A simple Streamlit app that analyses the algorithm used in a given ciphered database.")

tab1, tab2 = st.tabs(["📂 Upload a file", "⚡ Enter text manually"])

with tab1:
    uploaded_file = st.file_uploader("Upload your document", type=["pdf", "docx", "txt", "md"])

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
        placeholder="Enter your ciphered text here :",
        height=150
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

            svm_prediction, nn_prediction, svm_probabilities = classify_new_ciphertext(
                cipher_text, selector, scaler, svm_model, nn_model, label_encoder
            )
            st.header("Analysis Results:")
            st.write(f"Predicted Algorithm (SVM): {svm_prediction}")
            st.write(f"Predicted Algorithm (Neural Network): {nn_prediction}")
            
            # Display the probabilities in a tabular form
          # Sort the probabilities DataFrame in descending order by the probabilities
            probabilities_df = pd.DataFrame({
                'Algorithm': label_encoder.classes_,
                'Probability (SVM)': svm_probabilities
            }).sort_values(by='Probability (SVM)', ascending=False)

            st.dataframe(probabilities_df)
