import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from docx import Document

import fitz  # PyMuPDF
from transformers import BartForConditionalGeneration, BartTokenizer, pipeline
import textwrap
import tempfile

# Functions for file reading
def read_txt(file):
    return file.getvalue().decode("utf-8")

def read_docx(file):
    doc = Document(file)
    return " ".join([para.text for para in doc.paragraphs])

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc[page_num]
        text += page.get_text()
    doc.close()
    return text

def read_pdf(file):
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False)        
    # Write uploaded file content to the temporary file
    temp_file.write(file.read())
    # Close the temporary file to ensure changes are saved
    temp_file.close()
    # Get the file path of the temporary file
    file_path = temp_file.name

    return file_path, extract_text_from_pdf(file_path)

# Function for text summarization from pdf
def text_summarizer_from_pdf(pdf_path):
    pdf_text = extract_text_from_pdf(pdf_path)

    model_name = "facebook/bart-large-cnn"
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)

    inputs = tokenizer.encode("summarize: " + pdf_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    formatted_summary = "\n".join(textwrap.wrap(summary, width=80))
    return formatted_summary

# Summarizer pipeline for txt and docx files
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

st.title("Text Summarizer")
st.subheader("📁 Upload a pdf, docx or text file to generate a short summary")

# Sidebar to upload file
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["txt", "pdf", "docx"])
st.set_option('deprecation.showPyplotGlobalUse', False)

if uploaded_file:
    file_details = {"FileName:" : uploaded_file.name, "FileType:" : uploaded_file.type, "FileSize:" : uploaded_file.size}
    for key, value in file_details.items():
        st.sidebar.write(key, value)

    # Check the file type and read the file
    if uploaded_file.type == "text/plain":
        text = read_txt(uploaded_file)
    elif uploaded_file.type == "application/pdf":
        temp_path, text = read_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = read_docx(uploaded_file)
    else:
        st.error("File type not supported. Please upload a txt, pdf or docx file.")
        st.stop()

    # Generate summary
    if st.button('Generate Summary'):
        with st.spinner("Generating summary..."):
            try:
                if(uploaded_file.type == "application/pdf"):
                    pdf_file_path = temp_path
                    summary = text_summarizer_from_pdf(temp_path)
                    st.success(summary)
                else:
                    summary = summarizer(text, max_length=1000, min_length=30, do_sample=False)
                    st.success(summary[0]['summary_text'])
            except Exception as e:
                st.write(f"Failed to generate summary. Your file may have some problem. Please try again!")
