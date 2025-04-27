# Flask WebSocket Chat Application

A simple Flask application with WebSocket support using Flask-SocketIO for real-time communication.

## Features

- Real-time bidirectional communication using WebSockets
- Secure random secret key generation
- CORS enabled for cross-origin requests
- Gevent-based WebSocket server for better performance

## Prerequisites

- Python 3.x
- pip (Python package installer)

## Installation

1. Clone this repository or download the source code.

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Make sure you're in the project directory and your virtual environment is activated.

2. Run the application:
   ```bash
   python app.py
   ```

3. The server will start on `http://0.0.0.0:5000`

## Project Structure

- `app.py`: Main application file containing the Flask and SocketIO setup
- `requirements.txt`: List of Python package dependencies
- `venv/`: Virtual environment directory (created during setup)

## Dependencies

The project uses the following main packages:
- Flask: Web framework
- Flask-SocketIO: WebSocket support for Flask
- Gevent: Asynchronous networking library
- Gevent-WebSocket: WebSocket handler for Gevent

All dependencies are listed in `requirements.txt` with their specific versions.

## Security Notes

- The application generates a secure random secret key on each startup
- CORS is enabled for all origins (`*`) - modify this in production
- For production use, consider:
  - Setting up a persistent secret key using environment variables
  - Restricting CORS to specific origins
  - Using HTTPS
  - Implementing proper authentication

## License

This project is open source and available under the MIT License. 