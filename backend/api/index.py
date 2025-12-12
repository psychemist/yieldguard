import os
import sys

# Get the absolute path to the directory containing this file (backend/api)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory (backend)
backend_dir = os.path.dirname(current_dir)

# Add the backend directory to sys.path so 'src' and 'main' can be imported
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

# Import app from main
from main import app