from PyPDF2 import PdfReader
import io
import logging
from .embeddings import EmbeddingsService

logger = logging.getLogger(__name__)

class PDFService:
    def process_pdf(self, file):
        """
        Process a PDF file and extract its text content.
        
        Args:
            file: File object containing the PDF
            
        Returns:
            str: Extracted text from the PDF
            
        Raises:
            Exception: If there's an error processing the PDF
        """
        try:
            logger.debug("Starting PDF processing")
            
            # Check if file is empty
            if not file:
                raise Exception("Empty file provided")
                
            # Read the file content
            file_content = file.read()
            if not file_content:
                raise Exception("Empty file content")
                
            logger.debug("Creating PDF reader")
            pdf_reader = PdfReader(io.BytesIO(file_content))
            
            # Check if PDF is encrypted
            if pdf_reader.is_encrypted:
                raise Exception("Encrypted PDFs are not supported")
            
            # Extract text from all pages
            logger.debug("Extracting text from PDF")
            text = ""
            for i, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    else:
                        logger.warning(f"No text found on page {i+1}")
                except Exception as e:
                    logger.error(f"Error extracting text from page {i+1}: {str(e)}")
                    continue
            
            if not text.strip():
                raise Exception("No text could be extracted from the PDF")
            
            logger.debug("PDF processing completed successfully")
            # embeddings_service = EmbeddingsService()
            # embeddings = embeddings_service.custom_embed_text(text)
            # print(embeddings)
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
            raise Exception(f"Error processing PDF: {str(e)}") 