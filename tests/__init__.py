"""
Test package initialization.
Ensures the root directory is in the Python path.
"""
import os
import sys

# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add project root to Python path if not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("test module loaded.")