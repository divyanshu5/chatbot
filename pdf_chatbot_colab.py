# %% [markdown]
# # PDF Chatbot with Vector Database
# 
# This notebook implements a PDF chatbot that uses FAISS for efficient similarity search and OpenAI's GPT model for generating responses.
# 
# ## Setup Instructions
# 
# 1. Run all cells in sequence
# 2. Set your OpenAI API key when prompted
# 3. Upload a PDF file when the interface appears
# 4. Ask questions about the PDF content

# %%
# Install required packages
!pip install -q PyPDF2 openai sentence-transformers faiss-cpu gradio

# %%
# Import required libraries
import os
import PyPDF2
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import gradio as gr
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# %%
# Set your OpenAI API key
api_key = input("Enter your OpenAI API key: ")
os.environ['OPENAI_API_KEY'] = api_key
client = OpenAI()

# %%
class VectorDB:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384  # Dimension of the embeddings
        self.index = faiss.IndexFlatL2(self.dimension)
        self.text_chunks = []
        logger.debug("VectorDB initialized")

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> list:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
            
        logger.debug(f"Split text into {len(chunks)} chunks")
        return chunks

    def add_documents(self, text: str):
        """Add documents to the vector database."""
        try:
            # Split text into chunks
            chunks = self.chunk_text(text)
            self.text_chunks.extend(chunks)
            
            # Generate embeddings for chunks
            embeddings = self.model.encode(chunks)
            
            # Add to FAISS index
            self.index.add(embeddings.astype('float32'))
            
            logger.debug(f"Added {len(chunks)} chunks to vector database")
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}", exc_info=True)
            raise

    def search(self, query: str, k: int = 3) -> list:
        """Search for similar chunks in the vector database."""
        try:
            # Generate embedding for query
            query_embedding = self.model.encode([query])[0].astype('float32')
            
            # Search in FAISS
            distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
            
            # Get the chunks and their distances
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1:  # -1 indicates no result
                    results.append((self.text_chunks[idx], float(distances[0][i])))
            
            logger.debug(f"Found {len(results)} similar chunks")
            return results
            
        except Exception as e:
            logger.error(f"Error searching: {str(e)}", exc_info=True)
            raise

# %%
class PDFChatbot:
    def __init__(self):
        self.vector_db = VectorDB()
        self.messages = [
            {"role": "system", "content": "You are a helpful AI assistant that answers questions based on the provided PDF content. If the question is not related to the PDF content, politely inform the user that you can only answer questions about the PDF."}
        ]
        self.pdf_text = None
        
    def process_pdf(self, pdf_file):
        """Process the uploaded PDF file."""
        try:
            # Read PDF file
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            
            # Extract text from all pages
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            self.pdf_text = text
            
            # Add documents to vector database
            self.vector_db.add_documents(text)
            
            return "PDF processed successfully! You can now ask questions about the document."
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
            return f"Error processing PDF: {str(e)}"
    
    def get_response(self, user_message):
        """Get a response from the chatbot."""
        if not self.pdf_text:
            return "Please upload a PDF first."
        
        try:
            # Search for relevant chunks
            relevant_chunks = self.vector_db.search(user_message)
            
            # Format the context from relevant chunks
            context = "\n\n".join([chunk for chunk, _ in relevant_chunks])
            
            # Add user message and context to conversation history
            self.messages.append({
                "role": "user",
                "content": f"Context from PDF:\n{context}\n\nQuestion: {user_message}"
            })
            
            # Get response from OpenAI
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=self.messages,
                temperature=0.7
            )
            
            # Extract the response content
            ai_response = response.choices[0].message.content
            
            # Add AI response to conversation history
            self.messages.append({"role": "assistant", "content": ai_response})
            
            return ai_response
            
        except Exception as e:
            logger.error(f"Error getting response: {str(e)}", exc_info=True)
            return f"Sorry, I encountered an error: {str(e)}"

# %%
# Initialize the chatbot
chatbot = PDFChatbot()

# Create Gradio interface
def process_pdf_and_chat(pdf_file, user_message):
    if pdf_file is not None:
        result = chatbot.process_pdf(pdf_file)
        if "Error" in result:
            return result, ""
    
    if user_message:
        return "", chatbot.get_response(user_message)
    
    return "", ""

# Create the interface
iface = gr.Interface(
    fn=process_pdf_and_chat,
    inputs=[
        gr.File(label="Upload PDF"),
        gr.Textbox(label="Ask a question about the PDF")
    ],
    outputs=[
        gr.Textbox(label="Processing Status"),
        gr.Textbox(label="Chatbot Response")
    ],
    title="PDF Chatbot",
    description="Upload a PDF and ask questions about its content."
)

# Launch the interface
iface.launch(share=True) 