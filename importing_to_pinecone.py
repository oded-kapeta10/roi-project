import pandas as pd
from supabase import create_client
from pinecone import Pinecone
from openai import OpenAI
import os
from dotenv import load_dotenv

# 1. Supabase Credentials
# אנחנו מושכים את הערך לפי השם שהגדרת בקובץ ה-.env
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# 2. Pinecone Credentials
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "mental-health-index") # ניתן להוסיף ערך ברירת מחדל

# 3. LLMod (OpenAI Proxy) Credentials
LLMOD_API_KEY = os.getenv("LLMOD_API_KEY")
# כתובת ה-API שסופקה על ידי המרצה [cite: 115]
LLMOD_BASE_URL = os.getenv("LLMOD_BASE_URL", "https://api.llmod.ai/v1")

# --- Client Initialization ---
# Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize Pinecone client and target the specific index
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Initialize OpenAI client with the custom base URL (Proxy)
client = OpenAI(api_key=LLMOD_API_KEY, base_url=LLMOD_BASE_URL)


def get_embedding(text):
    """
    Generates a 1536-dimension vector for the given text using OpenAI's embedding model.
    """
    # Clean the text to ensure consistent formatting
    text = text.replace("\n", " ")

    # Request embedding from the model
    response = client.embeddings.create(
        input=[text],
        model= "RPRTHPB-text-embedding-3-small"
    )
    return response.data[0].embedding


def process_and_upload():
    """
    Fetches data from Supabase, generates embeddings for the 'Context' column,
    and uploads the vectors to Pinecone along with 'Response' metadata.
    """
    print("Step 1: Fetching data from Supabase 'experiment_table'...")
    # Fetching the first 1000 rows for the initial experiment
    response = supabase.table("experiment_table").select("*").execute()
    rows = response.data

    if not rows:
        print("No data found in Supabase.")
        return

    print(f"Step 2: Processing {len(rows)} rows (Generating embeddings and upserting)...")

    vectors_to_upsert = []

    for i, row in enumerate(rows):
        try:
            # We generate the vector based on the 'Context' (the user's question)
            # because that is what we will search against later.
            vector = get_embedding(row['Context'])

            # Prepare the Pinecone vector object
            vectors_to_upsert.append({
                "id": str(row['id']),  # Using Supabase ID as the unique identifier
                "values": vector,
                "metadata": {
                    "context": row['Context'],
                    "response": row['Response']  # Store the therapist answer as metadata
                }
            })

            # Batch upsert every 100 records to optimize performance
            if len(vectors_to_upsert) >= 100:
                index.upsert(vectors=vectors_to_upsert)
                vectors_to_upsert = []
                print(f"Progress: Uploaded {i + 1} records...")

        except Exception as e:
            print(f"Error processing row {i}: {e}")

    # Upload any remaining records
    if vectors_to_upsert:
        index.upsert(vectors=vectors_to_upsert)

    print("✅ Success! Your experiment data is now indexed in Pinecone.")


if __name__ == "__main__":
    process_and_upload()