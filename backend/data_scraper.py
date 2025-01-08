import requests
from bs4 import BeautifulSoup
import re

# Define headers to mimic a real browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
}

# Function to clean the extracted text (remove excessive newlines and whitespace)
def clean_text(text):
    # Remove multiple newlines or spaces
    text = re.sub(r'\n+', ' ', text)  # Replace multiple newlines with a single space
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = text.strip()  # Remove leading/trailing whitespace
    return text

# Scraping function
def scrape_website(url):
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Check if the request was successful
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract text and clean it
        text = soup.get_text(separator=" ")  # Use separator for better structure
        cleaned_text = clean_text(text)  # Clean the extracted text
        
        return cleaned_text  # Return the cleaned text content of the page
    except requests.exceptions.RequestException as e:
        print(f"Error scraping {url}: {e}")
        return None
