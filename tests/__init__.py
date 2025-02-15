#! /usr/bin/env python3
#tests/__init__.py
"""
Module: tests
Provides unit testing functionality for the tests module.
"""
import os
import sys

# Add the src directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

print("test module loaded.")