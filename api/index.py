import os
import sys

# Setup paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
backend_path = os.path.join(project_root, 'backend')

if backend_path not in sys.path:
    sys.path.append(backend_path)
if project_root not in sys.path:
    sys.path.append(project_root)

from backend.main import app
