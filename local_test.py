#!/usr/bin/env python3
"""
Simple Flask Test for AutoStatIQ
"""
import os
import sys

# Test imports one by one
print("🧪 Testing AutoStatIQ Local Deployment")
print("=" * 40)

# Test 1: Core Python libraries
try:
    import json
    import pandas as pd
    import numpy as np
    print("✅ Core Python & Data libraries")
except ImportError as e:
    print(f"❌ Core libraries: {e}")
    sys.exit(1)

# Test 2: Flask
try:
    from flask import Flask, render_template, request, jsonify
    print("✅ Flask web framework")
except ImportError as e:
    print(f"❌ Flask: {e}")
    sys.exit(1)

# Test 3: Advanced libraries
try:
    from factor_analyzer import FactorAnalyzer
    print("✅ Factor analyzer")
except ImportError as e:
    print(f"❌ Factor analyzer: {e}")

try:
    from lifelines import KaplanMeierFitter
    print("✅ Lifelines")
except ImportError as e:
    print(f"❌ Lifelines: {e}")

# Test 4: Import main app
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from app import app
    print("✅ AutoStatIQ app loads successfully")
except ImportError as e:
    print(f"❌ AutoStatIQ app: {e}")
    sys.exit(1)

# Test 5: Quick health check
try:
    with app.test_client() as client:
        response = client.get('/health')
        if response.status_code == 200:
            print("✅ Health endpoint working")
        else:
            print(f"❌ Health endpoint returned: {response.status_code}")
except Exception as e:
    print(f"❌ Health check failed: {e}")

print("=" * 40)
print("🎯 Local testing complete!")
print("\n🚀 Ready to start Flask server!")
print("Run: python app.py")
