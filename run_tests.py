#!/usr/bin/env python3
"""
AutoStatIQ Test Runner - Complete local testing
"""
import os
import sys
import subprocess
import time
import requests
import threading
from pathlib import Path

def test_dependencies():
    """Test all required dependencies"""
    print("🧪 Testing Dependencies...")
    print("=" * 50)
    
    # Core libraries
    try:
        import pandas as pd
        import numpy as np
        import scipy
        import matplotlib.pyplot as plt
        import plotly.express as px
        import seaborn as sns
        print("✅ Data Science Libraries (pandas, numpy, scipy, matplotlib, plotly, seaborn)")
    except ImportError as e:
        print(f"❌ Data Science Libraries: {e}")
        return False
    
    # Flask
    try:
        import flask
        from flask_cors import CORS
        print("✅ Flask Web Framework")
    except ImportError as e:
        print(f"❌ Flask: {e}")
        return False
    
    # Machine Learning
    try:
        import sklearn
        from sklearn.linear_model import LinearRegression
        print("✅ Scikit-learn Machine Learning")
    except ImportError as e:
        print(f"❌ Scikit-learn: {e}")
        return False
    
    # Advanced libraries
    try:
        from factor_analyzer import FactorAnalyzer
        print("✅ Factor Analyzer")
    except ImportError as e:
        print(f"⚠️  Factor Analyzer: {e} (May cause issues in advanced analysis)")
    
    try:
        from lifelines import KaplanMeierFitter
        print("✅ Lifelines Survival Analysis")
    except ImportError as e:
        print(f"⚠️  Lifelines: {e} (May cause issues in survival analysis)")
    
    # OpenAI
    try:
        import openai
        print("✅ OpenAI Integration")
    except ImportError as e:
        print(f"❌ OpenAI: {e}")
        return False
    
    return True

def start_flask_app():
    """Start Flask application in background"""
    try:
        print("\n🚀 Starting Flask Application...")
        print("=" * 50)
        
        # Start Flask in background
        process = subprocess.Popen(
            [sys.executable, "app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a bit for startup
        time.sleep(3)
        
        # Test if server is running
        try:
            response = requests.get("http://localhost:5000/health", timeout=5)
            if response.status_code == 200:
                print("✅ Flask server started successfully at http://localhost:5000")
                return process
            else:
                print(f"❌ Server health check failed: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"❌ Could not connect to server: {e}")
            return None
            
    except Exception as e:
        print(f"❌ Failed to start Flask app: {e}")
        return None

def test_endpoints():
    """Test key API endpoints"""
    print("\n🔍 Testing API Endpoints...")
    print("=" * 50)
    
    base_url = "http://localhost:5000"
    
    endpoints = [
        ("/health", "Health Check"),
        ("/", "Main Dashboard"),
        ("/api-docs", "API Documentation"),
        ("/api/info", "API Info")
    ]
    
    for endpoint, name in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=10)
            if response.status_code == 200:
                print(f"✅ {name}: {endpoint}")
            else:
                print(f"❌ {name}: {endpoint} (Status: {response.status_code})")
        except Exception as e:
            print(f"❌ {name}: {endpoint} (Error: {e})")

def test_file_upload():
    """Test file upload functionality"""
    print("\n📁 Testing File Upload...")
    print("=" * 50)
    
    # Check if sample data exists
    if Path("sample_data.csv").exists():
        print("✅ Sample CSV data file exists")
    else:
        print("❌ Sample CSV data file missing")
    
    if Path("sample_data.json").exists():
        print("✅ Sample JSON data file exists")
    else:
        print("❌ Sample JSON data file missing")

def main():
    """Main test runner"""
    print("🎯 AutoStatIQ Local Test Suite")
    print("=" * 60)
    
    # Test 1: Dependencies
    if not test_dependencies():
        print("\n❌ Dependency tests failed. Please install missing packages.")
        return
    
    # Test 2: Start Flask app
    flask_process = start_flask_app()
    if not flask_process:
        print("\n❌ Could not start Flask application.")
        return
    
    try:
        # Test 3: API endpoints
        test_endpoints()
        
        # Test 4: File upload readiness
        test_file_upload()
        
        print("\n" + "=" * 60)
        print("🎉 LOCAL TESTING COMPLETE!")
        print("=" * 60)
        print("✅ AutoStatIQ is ready for testing!")
        print("🌐 Open: http://localhost:5000")
        print("📊 Upload sample_data.csv to test analysis")
        print("❓ Try questions like:")
        print("   • 'Calculate average salary by department'")
        print("   • 'Show correlation between age and salary'")
        print("   • 'Create a visualization of salary distribution'")
        print("   • 'Perform regression analysis'")
        print("\n⚠️  Press Ctrl+C to stop the server")
        
        # Keep server running
        flask_process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping Flask server...")
        flask_process.terminate()
        flask_process.wait()
        print("✅ Server stopped successfully")
    
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        flask_process.terminate()

if __name__ == "__main__":
    main()
