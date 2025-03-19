import psycopg2
import numpy as np
import streamlit as st
import os
import hashlib
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Initialize Hugging Face MiniLM embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Connect to PostgreSQL
try:
    conn = psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB_NAME"),
        user=os.getenv("POSTGRES_DB_USER"),
        password=os.getenv("POSTGRES_DB_PASSWORD"),
        host=os.getenv("POSTGRES_DB_HOST"),
        port=os.getenv("POSTGRES_DB_PORT")
    )
    st.success("✅ Connected to PostgreSQL")
except Exception as e:
    st.error(f"❌ Error connecting to PostgreSQL: {e}")
    st.stop()

cursor = conn.cursor()

# Path to the "judgement" folder
judgement_folder = "C:/Users/21cs2/Desktop/original_dataset/original_dataset/IN-Abs/train-data/judgement"

def read_text_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except Exception as e:
        st.error(f"Error reading {file_path}: {e}")
        return None

# Function to generate embeddings using Hugging Face MiniLM
@st.cache_data  # Cache embeddings to avoid recomputation
def get_embedding(text):
    try:
        embedding = embedding_model.encode(text, convert_to_numpy=True).tolist()
        return embedding
    except Exception as e:
        st.error(f"Error generating embedding: {e}")
        return None

# Retrieve relevant files and display their content
st.header("🔍 Retrieve Relevant Case Content")
question_input = st.text_area("Enter your query:", height=150)

def get_relevant_files(question, top_n=5, min_similarity=0.5):
    try:
        question_embedding = get_embedding(question)
        if question_embedding is None:
            return []

        cursor.execute("SELECT file_name, embedding FROM document_embeddings;")
        rows = cursor.fetchall()

        if not rows:
            st.warning("No document embeddings found.")
            return []

        file_names, similarity_scores = [], []
        for row in rows:
            file_name, embedding_str = row[0], row[1][1:-1]  # Remove brackets
            embedding = np.fromstring(embedding_str, sep=",")

            if len(embedding) != len(question_embedding):  # Ensure matching dimensions
                st.error(f"Dimension mismatch for file {file_name}. Skipping.")
                continue

            similarity = np.dot(question_embedding, embedding) / (
                np.linalg.norm(question_embedding) * np.linalg.norm(embedding)
            )
            if similarity > min_similarity:
                file_names.append(file_name)
                similarity_scores.append(similarity)

        scored_files = sorted(zip(file_names, similarity_scores), key=lambda x: x[1], reverse=True)[:top_n]
        return [file for file, _ in scored_files]

    except Exception as e:
        st.error(f"Error retrieving relevant files: {e}")
        return []

if question_input:
    relevant_files = get_relevant_files(question_input)
    if relevant_files:
        st.write("✅ **Top Relevant Case Files:**")
        for file in relevant_files:
            file_path = os.path.join(judgement_folder, file)
            case_content = read_text_file(file_path)
            
            if case_content:
                st.subheader(f"📜 {file}")
                st.write(case_content[:1000] + "...")  # Show first 1000 characters
                with st.expander("Show Full Case Text"):
                    st.write(case_content)
    else:
        st.warning("No relevant cases found.")

# Close database connection
try:
    cursor.close()
    conn.close()
    st.success("Database connection closed successfully.")
except Exception as e:
    st.error(f"Error closing database connection: {e}")