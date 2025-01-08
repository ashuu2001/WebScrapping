from langchain.llms import Cohere as LangCohere
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryMemory
from backend.pinecone_utils import fetch_all_from_pinecone
from backend.embeddings import generate_embeddings  # Import the embedding function

def generate_response(query, index):
    # Generate the embedding (query vector) for the user query
    query_vector = generate_embeddings(query)
    
    # Fetch context from Pinecone using the query vector
    context = fetch_all_from_pinecone(index, query_vector)
    
    # Set up the language model and conversation chain
    llm = LangCohere(temperature=0.7, model="command-xlarge-nightly")
    chain = ConversationChain(llm=llm, memory=ConversationSummaryMemory(llm=llm))
    
    # Generate the response using the context and query
    response = chain.invoke(f"Context: {context}\nQuery: {query}")
    
    return response
