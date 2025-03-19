import psycopg2
import os
import hashlib
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Initialize Hugging Face MiniLM embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# PostgreSQL connection
try:
    conn = psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB_NAME"),
        user=os.getenv("POSTGRES_DB_USER"),
        password=os.getenv("POSTGRES_DB_PASSWORD"),
        host=os.getenv("POSTGRES_DB_HOST"),
        port=os.getenv("POSTGRES_DB_PORT")
    )
    cursor = conn.cursor()
    print("✅ Connected to PostgreSQL")
except Exception as e:
    print(f"❌ Error connecting to PostgreSQL: {e}")
    exit()

# Ensure pgvector extension is enabled
cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

# Create table if not exists
cursor.execute("""
    CREATE TABLE IF NOT EXISTS document_embeddings (
        id SERIAL PRIMARY KEY,
        file_name TEXT UNIQUE,
        file_hash TEXT UNIQUE,
        embedding VECTOR(384)  -- Match MiniLM output dimension
    );
""")
conn.commit()

# Path to the folder containing text files
judgement_folder = r"C:\Users\21cs2\Desktop\original_dataset\original_dataset\IN-Abs\train-data\judgement"

# Function to generate SHA-256 hash for file content
def get_file_hash(content):
    return hashlib.sha256(content.encode()).hexdigest()

# Function to read text file content
def read_text_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# Function to generate embeddings
def get_embedding(text):
    return embedding_model.encode(text, convert_to_numpy=True).tolist()

# Load all text file paths
file_paths = [os.path.join(judgement_folder, f) for f in os.listdir(judgement_folder) if f.endswith(".txt")]

# Insert unique files into the database
for file in file_paths:
    file_content = read_text_file(file)
    if not file_content:
        continue  # Skip if unable to read the file

    file_hash = get_file_hash(file_content)

    # Check if the file already exists in the database
    cursor.execute("SELECT 1 FROM document_embeddings WHERE file_hash = %s;", (file_hash,))
    if cursor.fetchone():
        print(f"⚠️ Skipping {file} (already in database)")
        continue

    # Generate embedding
    embedding = get_embedding(file_content)

    # Insert into the database
    try:
        cursor.execute(
            "INSERT INTO document_embeddings (file_name, file_hash, embedding) VALUES (%s, %s, %s::vector);",
            (os.path.basename(file), file_hash, embedding)
        )
        conn.commit()
        print(f"✅ Inserted {file} into the database")
    except Exception as e:
        print(f"Database Error: {e}")
        conn.rollback()

# Close database connection
cursor.close()
conn.close()
print("✅ Database connection closed.")
