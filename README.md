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

Installation
Clone the repository:

bash
Copy code
git clone https://github.com/Jatinnn27/EDD.git
cd ask-your-pdf
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Set up Google Generative AI credentials:

Place the service account JSON key file in the specified path.
Ensure the file has proper access to the Generative Language API with the required scopes.
Create a .env file to load environment variables:

env
Copy code
GOOGLE_API_KEY=<Your Google API Key>
Run the app:

bash
Copy code
streamlit run app.py
Usage
Upload PDFs:

Use the sidebar to upload one or multiple PDFs.
Click Process PDFs to extract and store the text.
Ask Questions:

Enter your question in the input box and click Submit.
The app retrieves the most relevant text segments and generates a detailed response.
Extract Images:

Click Extract Images in the sidebar to view images from the uploaded PDFs.
Key Components
Google Generative AI:

Provides embeddings for text processing and a conversational model for answering queries.
Requires appropriate service account credentials for access.
FAISS Vector Store:

Enables efficient semantic searches on large datasets of text chunks.
Streamlit:

Powers the interactive web app interface.
PyMuPDF (fitz):

Extracts images embedded within PDFs.
PyPDF2:

Handles PDF text extraction.
Customization
Styling: Modify the Streamlit layout and colors in the st.markdown block.
Prompt Template: Update the question-answering prompt for different use cases.
Chunking: Adjust chunk_size and chunk_overlap in RecursiveCharacterTextSplitter for different text granularity.
Limitations
Embedding Size: Large PDFs may require significant storage and computation for embeddings.
Model Access: Dependent on Google Generative AI availability and quotas.
PDF Structure: Inconsistent formatting in PDFs can affect text extraction quality.
Future Enhancements
Add support for more file formats (e.g., Word, Excel).
Integrate other embedding and QA models for comparison.
Implement a search bar for manual exploration of processed documents.

[Virtual Environment files (Too large to push in Github Repository)]
