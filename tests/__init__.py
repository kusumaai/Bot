"""
Test package initialization.
Ensures the root directory is in the Python path.
"""
import os
import sys

# Add the src directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

print("test module loaded.")