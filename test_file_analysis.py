import pandas as pd
import requests
import json

def test_file_upload_analysis():
    """Test file upload and analysis without OpenAI"""
    
    # Create sample data
    sample_data = {
        'Group': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
        'Score': [85, 92, 78, 94, 87, 96, 82, 79, 84],
        'Age': [25, 28, 32, 30, 35, 28, 33, 29, 31]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('test_data.csv', index=False)
    
    url = "http://127.0.0.1:5000/upload"
    
    try:
        print("Testing AutoStatIQ file upload analysis...")
        print("Uploading sample dataset...")
        
        with open('test_data.csv', 'rb') as f:
            files = {'file': ('test_data.csv', f, 'text/csv')}
            data = {'question': 'Compare scores across different groups'}
            
            response = requests.post(url, files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS! File analysis is working!")
            
            print("\n📊 Statistical Results:")
            if 'descriptive_stats' in result.get('results', {}):
                print("- Descriptive statistics calculated")
            if 't_test' in result.get('results', {}):
                print("- T-tests performed")
            if 'anova' in result.get('results', {}):
                print("- ANOVA analysis completed")
            if 'correlation_test' in result.get('results', {}):
                print("- Correlation analysis done")
                
            print(f"\n📈 Visualizations: {len(result.get('plots', {}))}")
            print(f"📄 Report: {result.get('report_filename', 'Generated')}")
            
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
    
    # Cleanup
    import os
    if os.path.exists('test_data.csv'):
        os.remove('test_data.csv')

if __name__ == "__main__":
    test_file_upload_analysis()
