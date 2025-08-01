#!/usr/bin/env python3
"""
AutoStatIQ Production Start Script
Optimized for Render cloud deployment
"""

import os
import sys
from pathlib import Path

def setup_production():
    """Setup production environment"""
    # Ensure directories exist
    directories = ['uploads', 'results', 'static/plots', 'logs']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Set production environment
    os.environ.setdefault('FLASK_ENV', 'production')
    
    print("🚀 AutoStatIQ Production Environment Setup Complete")
    print(f"📊 Environment: {os.environ.get('FLASK_ENV')}")
    print(f"🌍 Port: {os.environ.get('PORT', 5000)}")

if __name__ == "__main__":
    setup_production()
    
    # Import and run the app
    from app import app
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
