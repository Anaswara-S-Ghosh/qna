import psycopg2
import numpy as np
import streamlit as st
import google.generativeai as genai
import os
import hashlib
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

# Load environment variables from .env file
load_dotenv()

# Configure Google Gemini API Key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize the correct Gemini model for embeddings
embedding_model = genai.GenerativeModel("models/embedding-001")

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
        content TEXT,
        content_hash TEXT UNIQUE,  -- Use SHA-256 hash for uniqueness
        embedding VECTOR(768)      -- Adjust vector size based on Gemini embeddings
    );
""")
conn.commit()

# Path to the "judgement" folder
judgement_folder = "C:/Users/21cs2/Desktop/original_dataset/original_dataset/IN-Abs/train-data/judgement"

# Function to generate SHA-256 hash for content
def get_text_hash(text):
    """Generate a SHA-256 hash for the given text"""
    return hashlib.sha256(text.encode()).hexdigest()

# Function to read text files
def read_text_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# Load documents in parallel
def load_documents():
    files = [os.path.join(judgement_folder, f) for f in os.listdir(judgement_folder) if f.endswith(".txt")]
    with ThreadPoolExecutor(max_workers=4) as executor:
        documents = list(executor.map(read_text_file, files))
    return [doc for doc in documents if doc]  # Remove None values

documents = load_documents()

# Function to generate embeddings using Gemini API
def get_embedding(text):
    try:
        response = embedding_model.embed_content(text)
        return response["embedding"]  # Extract the embedding
    except Exception as e:
        st.error(f"Error generating embedding: {e}")
        return None

# Generate and store embeddings
st.header("🔄 Generating Embeddings...")

def store_embeddings(docs, batch_size=3):
    batches = [docs[i:i + batch_size] for i in range(0, len(docs), batch_size)]

    for batch in batches:
        with ThreadPoolExecutor(max_workers=4) as executor:
            embeddings = list(executor.map(get_embedding, batch))

        insert_data = [(doc, get_text_hash(doc), embedding) for doc, embedding in zip(batch, embeddings) if embedding]

        if insert_data:
            try:
                cursor.executemany(
                    "INSERT INTO document_chunks (content, content_hash, embedding) VALUES (%s, %s, %s) "
                    "ON CONFLICT (content_hash) DO NOTHING;",
                    insert_data
                )
                conn.commit()
            except Exception as e:
                st.error(f"Database Error: {e}")
                conn.rollback()  # Prevent database locks

store_embeddings(documents)

# Retrieve relevant chunks
st.header("Retrieve Relevant Chunks")
question_input = st.text_input("Enter your question:")

def get_relevant_chunks(question, top_n=5):
    try:
        question_embedding = get_embedding(question)
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
