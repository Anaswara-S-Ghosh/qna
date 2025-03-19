from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg2
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend's URL
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
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
    print("Connection to PostgreSQL successful")

    # Ensure pgvector extension is enabled
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id SERIAL PRIMARY KEY,
            content TEXT,
            embedding VECTOR(768)  -- Update dimension to match the model output (768 for MPNet)
        );
    """)
    conn.commit()
except Exception as e:
    raise Exception(f"Error connecting to PostgreSQL: {e}")

# Document chunks to be used
documents = [
    "Martinez v. Parker Properties (2019): Landlord evicted tenants claiming owner move-in, but rented to new tenants instead ",
    "Thompson v. Riverside Apartments (2018): Landlord falsely claimed major repairs needed for eviction",
    "Chen v. Metropolitan Housing (2020): Tenant evicted under false pretenses of building renovation",
    "Williams v. TechGiant Corp (2017): Company throttled device performance through updates",
    "Consumer Rights Group v. MobileWorld (2019): Planned obsolescence through software manipulation",
    "Johnson v. SmartPhone Inc (2016): Hidden performance degradation in older models"
]

# Initialize Hugging Face embeddings model (MPNet)
model_name = "sentence-transformers/all-mpnet-base-v2"
hf = HuggingFaceEmbeddings(model_name=model_name)
model = SentenceTransformer(model_name)

# Insert document chunks and embeddings into the database if not already present
for doc in documents:
    embedding = hf.embed_documents([doc])[0]
    cursor.execute("SELECT 1 FROM document_chunks WHERE content = %s LIMIT 1", (doc,))
    if cursor.fetchone() is None:
        cursor.execute(
            "INSERT INTO document_chunks (content, embedding) VALUES (%s, %s)",
            (doc, embedding)
        )
conn.commit()

# Define input schema for API request
class QueryInput(BaseModel):
    question: str

# Define route for user query
@app.post("/user_query")
def handle_user_query(query: QueryInput):
    try:
        question = query.question         # Generate embedding for the question
        question_embedding = hf.embed_query(question)

        # Retrieve all chunks and their embeddings from the database
        cursor.execute("SELECT content, embedding FROM document_chunks;")
        rows = cursor.fetchall()

        if not rows:
            raise HTTPException(status_code=404, detail="No document chunks found in the database.")

        # Calculate similarity scores
        chunks = []
        similarity_scores = []
        for row in rows:
            content = row[0]
            embedding = np.fromstring(row[1][1:-1], sep=",")  # Convert string to NumPy array
            similarity = np.dot(question_embedding, embedding) / (
                np.linalg.norm(question_embedding) * np.linalg.norm(embedding)
            )
            chunks.append(content)
            similarity_scores.append(similarity)

        # Combine chunks with their scores and sort by similarity
        scored_chunks = sorted(zip(chunks, similarity_scores), key=lambda x: x[1], reverse=True)

        # Set similarity threshold and retrieve relevant chunks
        threshold = 0.57
        relevant_chunks = [chunk for chunk, score in scored_chunks if score >= threshold]

        if not relevant_chunks:
            relevant_chunks = [scored_chunks[0][0]]  # Return top chunk if no chunks meet the threshold

        return {"question": question, "relevant_chunks": relevant_chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Close database connection on app shutdown
@app.on_event("shutdown")
def shutdown():
    try:
        cursor.close()
        conn.close()
        print("Database connection closed successfully.")
    except Exception as e:
        print(f"Error closing database connection: {e}")

# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000)
