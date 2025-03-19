import os
import numpy as np
import psycopg2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to PostgreSQL
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
    raise Exception(f"❌ Error connecting to PostgreSQL: {e}")

# Initialize embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Path to the "judgement" folder
judgement_folder = "C:/Users/21cs2/Desktop/original_dataset/original_dataset/IN-Abs/train-data/judgement"

def read_text_file(file_path):
    """Reads content from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except Exception:
        return None  # Return None if file can't be read

def get_embedding(text):
    """Generates embeddings using Hugging Face MiniLM."""
    try:
        return embedding_model.encode(text, convert_to_numpy=True).tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embedding: {e}")

# Input schema
class QueryInput(BaseModel):
    question: str

@app.post("/query_cases")
def query_cases(query: QueryInput):
    """Retrieves the top 5 most relevant case files based on query."""
    try:
        question_embedding = get_embedding(query.question)

        cursor.execute("SELECT file_name, embedding FROM document_embeddings;")
        rows = cursor.fetchall()

        if not rows:
            return {"message": "No document embeddings found."}

        # Compute similarity scores
        scored_files = []
        for row in rows:
            file_name, embedding_str = row[0], row[1][1:-1]  
            embedding = np.fromstring(embedding_str, sep=",")

            similarity = np.dot(question_embedding, embedding) / (
                np.linalg.norm(question_embedding) * np.linalg.norm(embedding)
            )

            scored_files.append((file_name, similarity))

        # Sort results by similarity (highest first) and take top 5
        scored_files.sort(key=lambda x: x[1], reverse=True)
        top_5_files = [file for file, score in scored_files[:5] if score >= 0.5]

        if not top_5_files:
            return {"message": "No relevant cases found."}

        # Retrieve file content
        cases = []
        for file in top_5_files:
            file_path = os.path.join(judgement_folder, file)
            content = read_text_file(file_path)
            if content:
                cases.append({"file": file, "content": content[:1000] + "..."})

        return {"query": query.question, "cases": cases}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Close database connection on shutdown
@app.on_event("shutdown")
def shutdown():
    cursor.close()
    conn.close()
    print("Database connection closed.")

# Run the API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
