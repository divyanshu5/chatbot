from openai import OpenAI
import os
import logging
from dotenv import load_dotenv
from .vector_db import VectorDB

logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Initialize OpenAI client with the API key
        os.environ["OPENAI_API_KEY"] = self.api_key
        self.client = OpenAI()
        
        # Initialize vector database
        self.vector_db = VectorDB()
        self.context = None
        self.messages = [
            {"role": "system", "content": "You are a helpful AI assistant that answers questions based on the provided PDF content. If the question is not related to the PDF content, politely inform the user that you can only answer questions about the PDF."}
        ]
        logger.debug("Chat service initialized")

    def initialize_chat(self, pdf_text):
        """
        Initialize the chat with PDF content as context.
        
        Args:
            pdf_text (str): Extracted text from the PDF
        """
        try:
            logger.debug("Initializing chat with PDF content")
            self.context = pdf_text
            
            # Add documents to vector database
            self.vector_db.add_documents(pdf_text)
            
            # Save the vector database
            self.vector_db.save_index()
            
            # Update system message
            self.messages[0]["content"] = f"""You are a helpful AI assistant that answers questions based on the provided PDF content.
The content has been processed and stored in a vector database for efficient retrieval.
If the question is not related to the PDF content, politely inform the user that you can only answer questions about the PDF."""
            
            logger.debug("Chat initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing chat: {str(e)}", exc_info=True)
            raise

    def get_response(self, user_message):
        """
        Get a response from the AI based on the user's message and PDF context.
        
        Args:
            user_message (str): The user's message
            
        Returns:
            str: The AI's response
        """
        try:
            if not self.context:
                return "Please upload a PDF first to start the conversation."
            
            logger.debug(f"Processing user message: {user_message}")
            
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
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=self.messages,
                temperature=0.7
            )
            
            # Extract the response content
            ai_response = response.choices[0].message.content
            
            # Add AI response to conversation history
            self.messages.append({"role": "assistant", "content": ai_response})
            
            logger.debug("Response generated successfully")
            return ai_response
            
        except Exception as e:
            logger.error(f"Error getting response: {str(e)}", exc_info=True)
            return f"Sorry, I encountered an error: {str(e)}" 