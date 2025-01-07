import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import fitz  # PyMuPDF
from PIL import Image
import io
from google.auth.transport.requests import Request
from google.oauth2 import service_account

# Path to the new service account key file
key_path = r"C:\Users\MAYANK\Documents\gen-lang-client-0577170796-eaf471f35885.json"

# Correct scopes for Generative Language API
scopes = ["https://www.googleapis.com/auth/generative-language"]

# Load and refresh credentials
credentials = service_account.Credentials.from_service_account_file(key_path, scopes=scopes)
credentials.refresh(Request())  # Refresh to fetch access tokens

# Configure Google Generative AI
genai.configure(api_key=None, credentials=credentials)

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    st.warning("Google API Key not found. Please check your environment variables.")

# Set page configuration
st.set_page_config(
    page_title="AskYourPDF 💬",
    page_icon="💭",
    layout="wide",  # Set the layout to wide for sidebar placement on the right
)

# Add custom styling for background and fonts
st.markdown(
    """
    <style>
        /* Background */
        body {
            background-color:rgb(207, 6, 6);
            color:rgb(249, 1, 1);
        }

        /* Main title */
        .stTitle {
            color:rgb(2, 34, 70);
            font-family: 'Arial Black', Gadget, sans-serif;
            font-size: 2.5em;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color:rgb(231, 9, 9);
            float: right;
        }

        /* Input fields */
        input {
            font-family: 'Verdana', Geneva, sans-serif;
            font-size: 1.1em;
        }

        /* Buttons */
        button {
            font-family: 'Verdana', Geneva, sans-serif;
            font-size: 1.1em;
            background-color:rgb(1, 112, 239);
            color: red;
            border: none;
            padding: 10px 15px;
            border-radius: 5px;
        }

        button:hover {
            background-color:rgb(2, 66, 121);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Utility Functions
def get_pdf_text(pdf_docs):
    """Extracts text from uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Splits text into manageable chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """Creates a FAISS vector store from text chunks."""
    embeddings = GoogleGenerativeAIEmbeddings(credentials=credentials, model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """Sets up the conversational chain with the AI model."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, respond with "Answer is not available in the context."\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(credentials=credentials, model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def load_faiss_index():
    """Loads a saved FAISS vector store."""
    embeddings = GoogleGenerativeAIEmbeddings(credentials=credentials, model="models/embedding-001")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

def extract_images_from_pdf(pdf_path):
    """Extracts images from a PDF."""
    images = []
    with fitz.open(pdf_path) as pdf:
        for page_index in range(len(pdf)):
            for img_index, img in enumerate(pdf[page_index].get_images(full=True)):
                xref = img[0]
                base_image = pdf.extract_image(xref)
                images.append((page_index + 1, img_index + 1, base_image["image"]))
    return images

def save_or_display_images(images):
    """Displays extracted images."""
    for page, index, image_bytes in images:
        image = Image.open(io.BytesIO(image_bytes))
        st.image(image, caption=f"Page {page}, Image {index}")

def user_input(user_question):
    """Handles user queries."""
    try:
        new_db = load_faiss_index()
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("Reply: ", response["output_text"])
    except Exception as e:
        st.error(f"An error occurred: {e}")

def main():
    """Main function for the Streamlit app."""
    st.title("AskYourPDF 💭")
    
    user_question = st.text_input("Ask a question about your uploaded PDFs:", placeholder="Type your question here...")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.header("📂 File Management")
        st.write("Upload and process your PDF files below.")
        pdf_docs = st.file_uploader("Upload PDF Files (multiple allowed)", accept_multiple_files=True)
        
        if st.button("Process PDFs"):
            if pdf_docs:
                with st.spinner("Processing your PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Your PDFs have been processed and are ready for questions!")
            else:
                st.warning("Please upload at least one PDF.")

        st.subheader("🖼 Extract Images")
        if st.button("Extract Images"):
            if pdf_docs:
                for pdf in pdf_docs:
                    images = extract_images_from_pdf(pdf)
                    save_or_display_images(images)
            else:
                st.warning("Please upload a PDF.")

if __name__ == "__main__":
    main()
