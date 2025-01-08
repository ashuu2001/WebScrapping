# Updated app.py
import streamlit as st
from backend.pinecone_utils import initialize_pinecone, fetch_all_from_pinecone, store_in_pinecone
from backend.data_scraper import scrape_website
from backend.chatbot_logic import generate_response
from frontend.upload_processing import process_uploaded_files
from backend.embeddings import generate_embeddings
from langchain.llms import Cohere as LangCohere
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryMemory
from streamlit_chat import message

# Streamlit UI Configuration
st.set_page_config(page_title="Chatbot", page_icon=":robot_face:")
st.title("Web & Files Chatbot 🤖")

# Initialize Pinecone and other configurations
pinecone_index = initialize_pinecone()

# Sidebar for Data Input
st.sidebar.header("Data Input")
uploaded_files = st.sidebar.file_uploader("Upload Files (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

# Web Scraping Input
st.sidebar.subheader("Web Scraping")
website_url = st.sidebar.text_input("Enter Website URL")
scrape_button = st.sidebar.button("Scrape Website")

# Scraping Functionality
if scrape_button and website_url:
    scraped_content = scrape_website(website_url)
    if scraped_content:
        store_in_pinecone(scraped_content, "scraped_data", pinecone_index)
        st.sidebar.success("Data from the website scraped and stored successfully!")
    else:
        st.sidebar.error("Failed to scrape data from the website.")

# File Upload Handling
if uploaded_files:
    content = process_uploaded_files(uploaded_files)
    for file in uploaded_files:
        store_in_pinecone(content, file.name, pinecone_index)
    st.sidebar.success("Uploaded files processed and stored successfully!")

# Initialize the conversation state
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = None
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'last_question' not in st.session_state:
    st.session_state['last_question'] = None
if 'last_response' not in st.session_state:
    st.session_state['last_response'] = None

# Function to handle chatbot responses
def getresponse(user_input):
    if isinstance(st.session_state['conversation'], list):
        st.session_state['conversation'] = None  # Reset if it's mistakenly set as a list

    if st.session_state['conversation'] is None:
        llm = LangCohere(temperature=0, model="command-xlarge-nightly")
        st.session_state['conversation'] = ConversationChain(
            llm=llm,
            verbose=True,
            memory=ConversationSummaryMemory(llm=llm)
        )

    # Check if the user is asking about their last question or response
    if user_input.lower() in ["what was my last question?", "last question?"]:
        if st.session_state['last_question']:
            return f"Your last question was: '{st.session_state['last_question']}'"
        else:
            return "You haven't asked any questions yet."
    
    if user_input.lower() in ["what was the output of my last question?", "last output?"]:
        if st.session_state['last_response']:
            return f"The output of your last question was: '{st.session_state['last_response']}'"
        else:
            return "I haven't provided any responses yet."

    # Save the current question as the last question
    st.session_state['last_question'] = user_input

    # Fetch context from Pinecone and integrate it into the user input
    context = fetch_all_from_pinecone(pinecone_index, generate_embeddings(user_input))
    if context:
        user_input = f"Context: {context}\nQuestion: {user_input}"

    # Generate response using the conversation model
    response = st.session_state['conversation'].predict(input=user_input)
    st.session_state['last_response'] = response  # Save the response as the last response
    return response

# Chat UI
response_container = st.container()
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("Your question goes here:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

        if submit_button:
            st.session_state['messages'].append(user_input)
            model_response = getresponse(user_input)
            st.session_state['messages'].append(model_response)

            with response_container:
                for i in range(len(st.session_state['messages'])):
                    if (i % 2) == 0:
                        message(st.session_state['messages'][i], is_user=True, key=str(i) + '_user')
                    else:
                        message(st.session_state['messages'][i], key=str(i) + '_AI')
