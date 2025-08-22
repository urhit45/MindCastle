#!/usr/bin/env python3
"""
Test script for TinyNet configuration
Run with: python test_config.py
"""

import sys
from pathlib import Path

# Add the config directory to the path
sys.path.insert(0, str(Path(__file__).parent / "config"))

try:
    from config import load_labels, LabelsConfig
    
    print("✅ Configuration loaded successfully!")
    
    # Load labels
    labels = load_labels()
    
    print(f"\n📊 Loaded Configuration:")
    print(f"Categories ({len(labels.categories)}): {labels.categories}")
    print(f"States ({len(labels.states)}): {labels.states}")
    print(f"Templates ({len(labels.templates)}): {labels.templates}")
    
    # Test the structure
    assert isinstance(labels.categories, list), "Categories should be a list"
    assert isinstance(labels.states, list), "States should be a list"
    assert isinstance(labels.templates, list), "Templates should be a list"
    
    print("\n✅ All assertions passed!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you have installed the requirements: pip install -r requirements.txt")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
