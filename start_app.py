#!/usr/bin/env python3
"""
Simple Flask Starter for AutoStatIQ Testing
"""
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    try:
        print("🚀 Starting AutoStatIQ Flask Application...")
        print("=" * 50)
        
        # Import and run app
        from app import app
        
        print("✅ App imported successfully")
        print("🌐 Starting server at http://localhost:5000")
        print("📊 Sample data files available:")
        print("   - sample_data.csv (Employee data)")
        print("   - sample_data.json (JSON format)")
        print("\n⚠️  Press Ctrl+C to stop the server")
        print("=" * 50)
        
        # Start the Flask app
        app.run(
            host='127.0.0.1',
            port=5000,
            debug=True,
            use_reloader=False
        )
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Try installing missing dependencies")
    except Exception as e:
        print(f"❌ Error starting app: {e}")

if __name__ == "__main__":
    main()
