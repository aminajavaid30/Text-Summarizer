import streamlit as st
from docx import Document
import fitz  # PyMuPDF
from transformers import pipeline

# Functions for file reading
def read_txt(file):
    return file.getvalue().decode("utf-8")

def read_docx(file):
    doc = Document(file)
    return " ".join([para.text for para in doc.paragraphs])

def extract_text_from_pdf(file):
    doc = fitz.open(file)
    text = ""
    for page_num in range(len(doc)):
        page = doc[page_num]
        text += page.get_text()
    doc.close()
    return text

# Summarizer pipeline for txt and docx files
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

st.title("Text Summarizer")
st.subheader("📁 Upload a pdf, docx or text file to generate a short summary")

# Sidebar to upload file
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["txt", "pdf", "docx"])

if uploaded_file:
    file_details = {"FileName:" : uploaded_file.name, "FileType:" : uploaded_file.type, "FileSize:" : uploaded_file.size}
    for key, value in file_details.items():
        st.sidebar.write(key, value)

    # Check the file type and read the file
    if uploaded_file.type == "text/plain":
        text = read_txt(uploaded_file)
    elif uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = read_docx(uploaded_file)
    else:
        st.error("File type not supported. Please upload a txt, pdf or docx file.")
        st.stop()

    # Generate summary
    if st.button('Generate Summary'):
        with st.spinner("Generating summary..."):
            try:
                summary = summarizer(text, max_length=1000, min_length=30, do_sample=False)
                st.success(summary[0]['summary_text'])
            except Exception as e:
                st.write("Failed to generate summary. Please try again!")
