from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import os
import logging
from .services.pdf_service import PDFService
from .services.chat_service import ChatService
from dotenv import load_dotenv
import markdown

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', os.urandom(24).hex())
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
socketio = SocketIO(app, cors_allowed_origins="*",async_mode='threading')

# Initialize services
pdf_service = PDFService()
chat_service = ChatService()

@app.route('/')
def index():
    return render_template('shadcn_chat.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are allowed'}), 400
    try:
        # Process PDF
        pdf_text = pdf_service.process_pdf(file)
        logger.debug("PDF processed successfully")
        # Initialize chat with PDF content
        chat_service.initialize_chat(pdf_text)
        logger.debug("Chat initialized with PDF content")
        return jsonify({'message': 'PDF processed successfully'})
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@socketio.on('connect')
def handle_connect():
    logger.debug('Client connected')
    emit('response', {'type': 'system', 'data': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    logger.debug('Client disconnected')

@socketio.on('message')
def handle_message(data):
    logger.debug(f"Received message: {data}")
    
    # Show typing indicator
    emit('response', {'type': 'typing', 'data': 'Server is typing...'})
    
    try:
        # Get response from chat service
        response = chat_service.get_response(data)
        # ai_md = response.choices[0].message.content  # Markdown text
        ai_html = markdown.markdown(response, extensions=['fenced_code', 'codehilite'])
        socketio.emit('response', {'type':'server', 'data': ai_html})

    except Exception as e:
        logger.error(f"Error processing message: {str(e)}", exc_info=True)
        socketio.emit('response', {'type': 'error', 'data': str(e)})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=3004, debug=True)