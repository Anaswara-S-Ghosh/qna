import psycopg2
import numpy as np
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import torch
import time
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

# Load environment variables from .env file
load_dotenv()

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

cursor = conn.cursor()

# Ensure pgvector extension is enabled
cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS document_chunks (
        id SERIAL PRIMARY KEY,
        content TEXT UNIQUE,
        embedding VECTOR(384)  -- 384 dimensions for MiniLM model
    );
""")
conn.commit()

# Path to the "judgement" folder
judgement_folder = "C:/Users/21cs2/Desktop/original_dataset/original_dataset/IN-Abs/train-data/judgement"
# GPU acceleration setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Use a smaller, faster model
@st.cache_resource  # Cache model loading to improve speed
def load_model():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    return SentenceTransformer(model_name, device=device)

model = load_model()
hf = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load documents in parallel
def read_text_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def load_documents():
    files = [os.path.join(judgement_folder, f) for f in os.listdir(judgement_folder) if f.endswith(".txt")]
    with ThreadPoolExecutor(max_workers=4) as executor:  # Use 4 threads for parallel reading
        documents = list(executor.map(read_text_file, files))
    return [doc for doc in documents if doc]  # Remove None values

documents = load_documents()

# Generate and store embeddings
st.header("Generating Embeddings...")

@st.cache_resource
def store_embeddings(docs):
    batch_size = 3
    batches = [docs[i:i + batch_size] for i in range(0, len(docs), batch_size)]

    for batch in batches:
        try:
            embeddings = hf.embed_documents(batch)  # Generate embeddings
            
            for doc, embedding in zip(batch, embeddings):
                cursor.execute("SELECT 1 FROM document_chunks WHERE content = %s LIMIT 1", (doc,))
                if cursor.fetchone() is None:
                    cursor.execute(
                        "INSERT INTO document_chunks (content, embedding) VALUES (%s, %s)",
                        (doc, embedding)
                    )
                    conn.commit()
        except Exception as e:
            st.error(f"Error generating embeddings: {e}")

store_embeddings(documents)

# Retrieve relevant chunks
st.header("Retrieve Relevant Chunks")
question_input = st.text_input("Enter your question:")

def get_relevant_chunks(question, top_n=5):
    try:
        question_embedding = hf.embed_query(question)
        cursor.execute("SELECT content, embedding FROM document_chunks;")
        rows = cursor.fetchall()

        if not rows:
            st.warning("No document chunks found.")
            return []

        chunks, similarity_scores = [], []
        for row in rows:
            content, embedding_str = row[0], row[1][1:-1]  # Remove brackets
            embedding = np.fromstring(embedding_str, sep=",")
            similarity = np.dot(question_embedding, embedding) / (
                np.linalg.norm(question_embedding) * np.linalg.norm(embedding)
            )
            chunks.append(content)
            similarity_scores.append(similarity)

        scored_chunks = sorted(zip(chunks, similarity_scores), key=lambda x: x[1], reverse=True)[:top_n]

        return [chunk for chunk, _ in scored_chunks]

    except Exception as e:
        st.error(f"Error retrieving relevant chunks: {e}")
        return []

if question_input:
    relevant_chunks = get_relevant_chunks(question_input)
    if relevant_chunks:
        st.write("✅ **Top Relevant Chunks Found:**")
        st.write(relevant_chunks)
    else:
        st.warning("No relevant chunks found.")

# Close database connection
try:
    cursor.close()
    conn.close()
    st.success("Database connection closed successfully.")
except Exception as e:
    st.error(f"Error closing database connection: {e}")
