# Production Server for Face Extraction Service
# Uses Waitress (WSGI Server) for robust performance on Windows
from waitress import serve
from face_extraction_app import app
import logging

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('waitress')
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  EDU-BUS FACE EXTRACTION SERVICE (PRODUCTION)")
    print("  🚀 Powered by Waitress & InsightFace")
    print("="*50)
    print("Host: 0.0.0.0")
    print("Port: 5001")
    print("Threads: 4")
    print("="*50 + "\n")
    
    # Run Waitress
    serve(app, host='0.0.0.0', port=5001, threads=4)
