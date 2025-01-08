from PyPDF2 import PdfReader
from docx import Document

def process_uploaded_files(files):
    content = ""
    for file in files:
        ext = file.name.split('.')[-1].lower()
        if ext == "pdf":
            reader = PdfReader(file)
            for page in reader.pages:
                content += page.extract_text()
        elif ext == "docx":
            doc = Document(file)
            content += "\n".join([p.text for p in doc.paragraphs])
        elif ext == "txt":
            content += file.read().decode("utf-8")
    return content
