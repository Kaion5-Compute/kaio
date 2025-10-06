#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test script to verify the package imports correctly."""

import sys
import os

# Add current directory to Python path for testing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from kaio import Client
    print("Success: Successfully imported Client from kaio")
    
    # Test client initialization
    client = Client("https://api.kaion5.com")
    print("Success: Successfully created Client instance")
    print(f"Success: Client API base: {client.api_base}")
    
except ImportError as e:
    print(f"Error: Import failed: {e}")
except Exception as e:
    print(f"Error: {e}")