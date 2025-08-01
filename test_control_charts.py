import requests
import pandas as pd
import json

def test_control_chart_analysis():
    """Test the new control chart functionality"""
    
    # Create quality control test data
    quality_data = {
        'Sample': list(range(1, 31)),
        'Measurement': [23.5, 24.1, 23.8, 24.2, 23.9, 24.0, 23.7, 24.3, 23.6, 24.1,
                       23.8, 24.0, 23.9, 24.2, 23.7, 24.1, 23.5, 24.0, 23.8, 24.3,
                       23.9, 24.1, 23.6, 24.0, 23.8, 24.2, 23.7, 24.1, 23.9, 24.0],
        'Defects': [2, 1, 3, 0, 2, 1, 4, 2, 1, 3, 2, 1, 2, 0, 3, 1, 2, 4, 1, 2,
                   3, 1, 2, 0, 1, 2, 3, 1, 2, 1],
        'Sample_Size': [50] * 30
    }
    
    df = pd.DataFrame(quality_data)
    df.to_csv('test_control_charts.csv', index=False)
    
    url = "http://127.0.0.1:5000/upload"
    
    try:
        print("🔧 Testing AutoStatIQ Control Chart Analysis...")
        print("📊 Uploading quality control dataset...")
        
        with open('test_control_charts.csv', 'rb') as f:
            files = {'file': ('test_control_charts.csv', f, 'text/csv')}
            data = {'question': 'Analyze this quality control data with control charts for process monitoring'}
            
            response = requests.post(url, files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS! Control Chart Analysis is working!")
            
            print("\n📊 Analysis Results:")
            results = result.get('results', {})
            
            # Check for SPC analysis
            if results.get('spc_analysis'):
                print("✅ SPC Analysis detected and performed")
                
                # Check for control charts
                control_charts = results.get('control_charts', {})
                if control_charts:
                    print(f"📈 Control Charts Generated: {len(control_charts)}")
                    for chart_name, chart_data in control_charts.items():
                        chart_type = chart_data.get('type', 'Unknown')
                        column = chart_data.get('column', 'Unknown')
                        print(f"  - {chart_type} for {column}")
                        
                        # Check statistics
                        stats = chart_data.get('statistics', {})
                        if 'center_line' in stats and 'ucl' in stats:
                            print(f"    Center Line: {stats['center_line']:.4f}")
                            print(f"    UCL: {stats['ucl']:.4f}")
                            ooc_points = stats.get('out_of_control_points', [])
                            if ooc_points:
                                print(f"    ⚠️  Out-of-Control Points: {len(ooc_points)}")
                            else:
                                print(f"    ✅ Process In Control")
                
                # Check for interpretations
                interpretations = results.get('interpretations', {})
                if interpretations:
                    print(f"📝 Control Chart Interpretations: {len(interpretations)}")
                
                # Check for recommendations
                recommendations = results.get('recommendations', {})
                if recommendations.get('recommended_charts'):
                    recommended_count = len(recommendations['recommended_charts'])
                    print(f"💡 Chart Recommendations: {recommended_count}")
                    
            else:
                print("ℹ️  No SPC analysis triggered (check keywords or data patterns)")
            
            # Check regular analysis
            if 'descriptive_stats' in results:
                print("✅ Descriptive statistics calculated")
            if 't_test' in results:
                print("✅ Statistical tests performed")
                
            print(f"\n📈 Total Visualizations: {len(result.get('plots', {}))}")
            if results.get('control_charts'):
                print(f"📊 Control Charts: {len(results['control_charts'])}")
            
            print(f"📄 Report Generated: {result.get('report_filename', 'Yes')}")
            print(f"🤖 AI Interpretation: {'Available' if result.get('interpretation') else 'Not available'}")
            
        else:
            print(f"❌ FAILED! Status code: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        
    finally:
        # Cleanup
        import os
        try:
            os.remove('test_control_charts.csv')
        except:
            pass

def test_control_chart_keywords():
    """Test control chart detection with SPC keywords"""
    
    url = "http://127.0.0.1:5000/api/analyze"
    
    test_questions = [
        "How do I create control charts for my manufacturing process?",
        "What are X-bar and R charts used for in quality control?",
        "Help me analyze process capability with SPC methods",
        "I need to monitor my process with control charts",
        "What control chart should I use for defect counts?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        try:
            print(f"\n🔍 Test {i}: Testing SPC keyword detection...")
            print(f"Question: {question}")
            
            payload = {"question": question}
            response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
            
            if response.status_code == 200:
                result = response.json()
                analysis = result.get('results', {}).get('text_analysis', {})
                analysis_type = analysis.get('analysis_type', 'unknown')
                recommended_tests = analysis.get('recommended_tests', [])
                
                print(f"✅ Analysis Type: {analysis_type}")
                print(f"📋 Recommended Tests: {', '.join(recommended_tests) if recommended_tests else 'None'}")
                
                # Check if SPC/control chart concepts are mentioned
                interpretation = result.get('interpretation', '').lower()
                if any(keyword in interpretation for keyword in ['control chart', 'spc', 'process control']):
                    print("✅ Control chart concepts detected in interpretation")
                else:
                    print("ℹ️  Control chart concepts not prominently mentioned")
                    
            else:
                print(f"❌ Failed with status: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    print("🚀 AutoStatIQ Control Chart Testing Suite")
    print("=" * 50)
    
    test_control_chart_analysis()
    
    print("\n" + "=" * 50)
    print("🔍 Testing SPC Keyword Detection")
    print("=" * 50)
    
    test_control_chart_keywords()
    
    print("\n✨ Testing Complete!")
