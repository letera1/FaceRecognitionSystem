"""Application entry point"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.app import app
from src.config import Config

if __name__ == '__main__':
    print(f"🚀 Starting Face Recognition System")
    print(f"📍 Server: http://{Config.HOST}:{Config.PORT}")
    print(f"📊 Model: {Config.EMBEDDING_MODEL}")
    
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG
    )
