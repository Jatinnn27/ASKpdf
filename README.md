README: AskYourPDF 💬
Overview
AskYourPDF is a Streamlit-based web application designed to allow users to interact with their PDF documents using natural language. It extracts text and images from PDFs, processes the content, and answers user queries using Google Generative AI embeddings and vector search.

Features
PDF Text Extraction: Extract text from multiple PDF files.
Text Chunking: Splits extracted text into manageable chunks for processing.
Vector Store: Utilizes FAISS for efficient storage and retrieval of embeddings.
Generative AI QA: Leverages Google Generative AI for answering queries about the uploaded documents.
Image Extraction: Extracts and displays images from PDF documents.
Interactive UI: Easy-to-use interface for uploading files, asking questions, and extracting images.
How It Works
Text Processing Pipeline:

PDFs are uploaded, and text is extracted using PyPDF2.
Text is split into chunks using RecursiveCharacterTextSplitter.
Chunks are stored in a FAISS vector database for efficient similarity search.
Embeddings:

Google Generative AI embeddings are used to transform text chunks into numerical vectors.
These embeddings enable semantic similarity searches, helping to retrieve relevant document segments for answering questions.
Question Answering:

Queries are handled using a custom prompt with the Google Generative AI ChatGoogleGenerativeAI model.
The model generates detailed responses based on retrieved document chunks.
Image Extraction:

Images are extracted from PDFs using PyMuPDF (fitz), enabling visualization of embedded media.

[Virtual Environment files (Too large to push in Github Repository)]
