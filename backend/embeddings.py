import cohere
import os
from dotenv import load_dotenv

load_dotenv()

cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))

# def generate_embeddings(text):
#     response = cohere_client.embed(texts=[text], model="embed-english-v2.0", truncate="END")
#     return response.embeddings[0]
def generate_embeddings(text):
    response = cohere_client.embed(texts=[text], model="embed-english-v2.0", truncate="END")
    embedding = response.embeddings[0]
    return embedding