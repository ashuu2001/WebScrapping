from pinecone import Pinecone, ServerlessSpec
from backend.embeddings import generate_embeddings  # Adjust the import according to your file structure
import os
from dotenv import load_dotenv

load_dotenv()

def initialize_pinecone():
    # Create a Pinecone instance
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    index_name = "vkaps-data"

    # Check if the index exists; if not, create it
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=4096,  # Adjust based on your embedding size
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")  # Adjust region if needed
        )

    # Return the initialized index
    return pc.Index(index_name)

def store_in_pinecone(text, doc_id, index):
    embedding = generate_embeddings(text)  # Import from embeddings.py
    index.upsert([(doc_id, embedding, {"text": text[:5000000]})])

# def fetch_all_from_pinecone(index):
#     query_response = index.query(vector=[0]*768, top_k=100, include_metadata=True)
#     return "\n".join(match.metadata.get("content", "") for match in query_response.matches)
def fetch_all_from_pinecone(index, query_vector):
    # Perform the query with the vector
    query_result = index.query(
        vector=query_vector,  # Use the query vector here
        top_k=5,  # Adjust based on how many results you want
        include_metadata=True
    )
    
    # Print the query result to inspect the structure
    print(query_result)  # Debugging: Print the full query result
    
    # Check if 'matches' exists in the result and inspect the metadata structure
    if 'matches' in query_result:
        for match in query_result['matches']:
            print(match['metadata'])  # Print metadata for each match to inspect its structure
    
    # Fetch context (ensure you handle missing keys safely)
    context = [
        match['metadata'].get('text', 'No content available')  # Use .get to handle missing keys
        for match in query_result.get('matches', [])
    ]
    
    # Filter out irrelevant content like 'Mod_Security' error
    filtered_context = [text for text in context if 'Mod_Security' not in text]
    
    return "\n".join(filtered_context) if filtered_context else "No relevant content found."
