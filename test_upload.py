#!/usr/bin/env python3
"""
Quick test script to verify file upload functionality
"""

import os
import sys
import requests
import json

def test_file_upload():
    """Test the file upload endpoint"""
    
    # Check if app is running
    try:
        response = requests.get('http://localhost:5000')
        print("✅ App is running on localhost:5000")
    except requests.exceptions.ConnectionError:
        print("❌ App is not running. Please start the app first with: python app.py")
        return False
    
    # Test file upload with sample data
    sample_file = 'sample_data.csv'
    if not os.path.exists(sample_file):
        print(f"❌ Sample file {sample_file} not found")
        return False
    
    # Upload file
    try:
        with open(sample_file, 'rb') as f:
            files = {'file': (sample_file, f, 'text/csv')}
            data = {'question': 'What is the correlation between age and salary?'}
            response = requests.post('http://localhost:5000/upload', files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✅ File upload successful!")
                print(f"✅ Analysis completed")
                print(f"✅ Report generated: {result.get('report_filename', 'N/A')}")
                return True
            else:
                print(f"❌ Upload failed: {result.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ HTTP error: {response.status_code}")
            print(f"Response: {response.text[:200]}...")
            return False
            
    except Exception as e:
        print(f"❌ Error during upload: {str(e)}")
        return False

def test_directories():
    """Test that required directories exist"""
    required_dirs = ['uploads', 'results', 'static/plots', 'logs', 'temp']
    
    print("Testing directory structure...")
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}/ exists")
        else:
            print(f"❌ {dir_path}/ missing")
            return False
    
    return True

def main():
    print("🔍 Testing AutoStatIQ File Upload Functionality")
    print("=" * 50)
    
    # Test directories
    if not test_directories():
        print("\n❌ Directory structure test failed")
        return
    
    # Test file upload
    if test_file_upload():
        print("\n✅ All tests passed! File upload is working correctly.")
    else:
        print("\n❌ File upload test failed. Check the logs for more details.")

if __name__ == "__main__":
    main()
