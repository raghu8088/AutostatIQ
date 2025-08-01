import requests
import json

# Test the AutoStatIQ API with a simple statistical question
def test_question_analysis():
    url = "http://127.0.0.1:5000/api/analyze"
    
    test_question = "What is the difference between mean and median, and when should I use each one?"
    
    payload = {
        "question": test_question
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        print("Testing AutoStatIQ API with OpenAI integration...")
        print(f"Question: {test_question}")
        print("\nSending request...")
        
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS! OpenAI integration is working!")
            print(f"\nAnalysis Type: {result['results']['text_analysis']['analysis_type']}")
            print(f"Recommended Tests: {result['results']['text_analysis']['recommended_tests']}")
            print(f"\nInterpretation: {result['interpretation'][:200]}...")
            print(f"\nConclusion: {result['conclusion'][:200]}...")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Connection error: {e}")

if __name__ == "__main__":
    test_question_analysis()
