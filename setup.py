#! /usr/bin/env python3
#setup.py
"""
Module: setup
Provides setup functionality for the Bot.
""" 
from setuptools import setup, find_packages

setup(
    name="Bot",
    version="0.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)