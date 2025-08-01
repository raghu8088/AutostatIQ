"""
AutoStatIQ Python Client Library
Ready-to-use Python SDK for AutoStatIQ API integration

Author: Roman Chaudhary
Contact: chaudharyroman.com.np
Version: 1.0.0
"""

import requests
import json
from typing import Optional, Dict, Any, Union, List
from pathlib import Path
import time
import warnings

class AutoStatIQClient:
    """
    Python client for AutoStatIQ statistical analysis API
    
    This client provides a simple interface to interact with the AutoStatIQ API
    for comprehensive statistical analysis. It supports file uploads, natural
    language questions, and report generation.
    
    Example usage:
        client = AutoStatIQClient()
        results = client.analyze_file("data.csv", "Perform correlation analysis")
        client.download_report(results['report_url'], "report.docx")
    """
    
    def __init__(self, base_url: str = "http://127.0.0.1:5000", api_key: Optional[str] = None, timeout: int = 300):
        """
        Initialize AutoStatIQ client
        
        Args:
            base_url: Base URL of AutoStatIQ API (default: http://127.0.0.1:5000)
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds (default: 300)
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({'X-API-Key': api_key})
        
        # Set user agent
        self.session.headers.update({
            'User-Agent': 'AutoStatIQ-Python-Client/1.0.0'
        })
    
    def check_health(self) -> Dict[str, Any]:
        """
        Check API health and configuration status
        
        Returns:
            Dictionary containing health status and API information
            
        Raises:
            Exception: If health check fails
        """
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise Exception(f"Health check failed: {str(e)}")
    
    def get_api_info(self) -> Dict[str, Any]:
        """
        Get comprehensive API information and capabilities
        
        Returns:
            Dictionary containing API info, supported methods, and features
            
        Raises:
            Exception: If API info request fails
        """
        try:
            response = self.session.get(f"{self.base_url}/api/info", timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise Exception(f"Failed to get API info: {str(e)}")
    
    def analyze_file(self, file_path: Union[str, Path], question: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform statistical analysis on uploaded file
        
        Args:
            file_path: Path to the file to analyze (CSV, XLSX, DOCX, PDF, JSON, TXT)
            question: Optional natural language statistical question
            
        Returns:
            Dictionary containing comprehensive analysis results including:
            - success: Boolean indicating if analysis succeeded
            - results: Statistical analysis results
            - interpretation: AI-generated interpretation
            - conclusion: Professional conclusions
            - plots: Generated visualizations
            - report_filename: Generated report filename
            
        Raises:
            FileNotFoundError: If the specified file doesn't exist
            Exception: If analysis fails
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Validate file format
        supported_extensions = {'.csv', '.xlsx', '.xls', '.docx', '.pdf', '.json', '.txt'}
        if file_path.suffix.lower() not in supported_extensions:
            raise ValueError(f"Unsupported file format. Supported: {', '.join(supported_extensions)}")
        
        try:
            with open(file_path, 'rb') as file:
                files = {'file': (file_path.name, file, self._get_mime_type(file_path))}
                data = {'question': question} if question else {}
                
                response = self.session.post(
                    f"{self.base_url}/upload",
                    files=files,
                    data=data,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
        except requests.RequestException as e:
            raise Exception(f"Analysis failed: {str(e)}")
    
    def analyze_question(self, question: str) -> Dict[str, Any]:
        """
        Analyze a natural language statistical question without file upload
        
        Args:
            question: Statistical question in natural language
            
        Returns:
            Dictionary containing analysis recommendations and interpretation
            
        Raises:
            Exception: If analysis fails
        """
        try:
            response = self.session.post(
                f"{self.base_url}/api/analyze",
                json={'question': question},
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise Exception(f"Question analysis failed: {str(e)}")
    
    def download_report(self, report_url_or_id: str, save_path: Union[str, Path]) -> Path:
        """
        Download generated statistical analysis report
        
        Args:
            report_url_or_id: Report URL (e.g., "/download/report_123") or report ID
            save_path: Path to save the downloaded report
            
        Returns:
            Path to the saved report file
            
        Raises:
            Exception: If download fails
        """
        save_path = Path(save_path)
        
        # Extract report ID from URL if full URL provided
        if report_url_or_id.startswith('/download/'):
            report_id = report_url_or_id.split('/')[-1]
        elif report_url_or_id.startswith('http'):
            report_id = report_url_or_id.split('/')[-1]
        else:
            report_id = report_url_or_id
        
        try:
            response = self.session.get(
                f"{self.base_url}/download/{report_id}",
                timeout=60
            )
            response.raise_for_status()
            
            # Create directory if it doesn't exist
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save file
            with open(save_path, 'wb') as file:
                file.write(response.content)
            
            return save_path
        except requests.RequestException as e:
            raise Exception(f"Download failed: {str(e)}")
    
    def batch_analyze(self, file_paths: List[Union[str, Path]], 
                     questions: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Analyze multiple files in batch
        
        Args:
            file_paths: List of file paths to analyze
            questions: Optional list of questions corresponding to each file
            
        Returns:
            List of analysis results for each file
        """
        results = []
        questions = questions or [None] * len(file_paths)
        
        for i, (file_path, question) in enumerate(zip(file_paths, questions)):
            try:
                print(f"Analyzing file {i+1}/{len(file_paths)}: {file_path}")
                result = self.analyze_file(file_path, question)
                results.append(result)
                
                # Add small delay between requests to be respectful
                if i < len(file_paths) - 1:
                    time.sleep(1)
                    
            except Exception as e:
                results.append({
                    "success": False, 
                    "error": str(e), 
                    "file": str(file_path)
                })
        
        return results
    
    def _get_mime_type(self, file_path: Path) -> str:
        """Get MIME type based on file extension"""
        extension = file_path.suffix.lower()
        mime_types = {
            '.csv': 'text/csv',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.xls': 'application/vnd.ms-excel',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.pdf': 'application/pdf',
            '.json': 'application/json',
            '.txt': 'text/plain'
        }
        return mime_types.get(extension, 'application/octet-stream')
    
    def __repr__(self) -> str:
        return f"AutoStatIQClient(base_url='{self.base_url}')"


# Convenience functions for quick usage
def quick_analysis(file_path: str, question: str = None, 
                  base_url: str = "http://127.0.0.1:5000") -> Dict[str, Any]:
    """
    Quick one-line analysis function
    
    Args:
        file_path: Path to file to analyze
        question: Optional statistical question
        base_url: API base URL
        
    Returns:
        Analysis results dictionary
    """
    client = AutoStatIQClient(base_url)
    return client.analyze_file(file_path, question)


def batch_analysis(file_paths: List[str], questions: List[str] = None, 
                  base_url: str = "http://127.0.0.1:5000") -> List[Dict[str, Any]]:
    """
    Analyze multiple files in batch
    
    Args:
        file_paths: List of file paths to analyze
        questions: Optional list of questions for each file
        base_url: API base URL
        
    Returns:
        List of analysis results
    """
    client = AutoStatIQClient(base_url)
    return client.batch_analyze(file_paths, questions)


def check_api_status(base_url: str = "http://127.0.0.1:5000") -> bool:
    """
    Quick check if AutoStatIQ API is available and healthy
    
    Args:
        base_url: API base URL
        
    Returns:
        True if API is healthy, False otherwise
    """
    try:
        client = AutoStatIQClient(base_url)
        health = client.check_health()
        return health.get('status') == 'healthy'
    except Exception:
        return False


# Example usage and testing
if __name__ == "__main__":
    # Example usage demonstration
    print("AutoStatIQ Python Client Demo")
    print("=" * 40)
    
    # Initialize client
    client = AutoStatIQClient()
    
    # Check if API is available
    try:
        health = client.check_health()
        print(f"✅ API Status: {health['status']}")
        print(f"📊 Version: {health['version']}")
        print(f"🔧 Features: {', '.join(health['features'])}")
        print()
    except Exception as e:
        print(f"❌ API not available: {e}")
        print("Please ensure AutoStatIQ server is running on http://127.0.0.1:5000")
        exit(1)
    
    # Get API information
    try:
        api_info = client.get_api_info()
        print(f"📈 Statistical Methods: {len(api_info['statistical_methods'])}")
        print(f"📁 Supported Formats: {', '.join(api_info['supported_formats'])}")
        print(f"📊 Control Charts: {len(api_info['control_chart_types'])}")
        print()
    except Exception as e:
        print(f"⚠️ Could not get API info: {e}")
    
    # Example analysis (you would replace this with your actual data file)
    example_question = "Perform comprehensive statistical analysis with correlations and hypothesis testing"
    
    print("To use this client with your data:")
    print("1. client = AutoStatIQClient()")
    print("2. results = client.analyze_file('your_data.csv', 'your_question')")
    print("3. client.download_report(results['report_filename'], 'report.docx')")
    print()
    
    print("Example for natural language analysis:")
    print("client.analyze_question('What statistical test should I use for comparing two groups?')")
    print()
    
    print("For batch processing:")
    print("results = client.batch_analyze(['file1.csv', 'file2.xlsx'], ['question1', 'question2'])")
    print()
    
    print("💡 This client supports all AutoStatIQ features:")
    print("   • 25+ Statistical Methods")
    print("   • Control Charts (SPC)")
    print("   • Advanced Analytics (PCA, Clustering)")
    print("   • Professional Report Generation")
    print("   • Intelligent Interpretations")
    print("   • Multi-format File Support")
