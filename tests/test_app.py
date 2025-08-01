import pytest
import os
import json
import tempfile
from app import app, StatisticalAnalyzer, ReportGenerator
import pandas as pd

@pytest.fixture
def client():
    """Create a test client for the Flask application."""
    app.config['TESTING'] = True
    app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
    app.config['WTF_CSRF_ENABLED'] = False
    
    with app.test_client() as client:
        with app.app_context():
            yield client

@pytest.fixture
def sample_csv_data():
    """Create sample CSV data for testing."""
    return pd.DataFrame({
        'group': ['A', 'A', 'B', 'B', 'A', 'B'],
        'value1': [1, 2, 3, 4, 5, 6],
        'value2': [10, 20, 30, 40, 50, 60]
    })

class TestStatisticalAnalyzer:
    """Test cases for StatisticalAnalyzer class."""
    
    def test_descriptive_analysis(self, sample_csv_data):
        """Test descriptive analysis functionality."""
        analyzer = StatisticalAnalyzer()
        analyzer.data = sample_csv_data
        
        results = analyzer.perform_descriptive_analysis(sample_csv_data)
        
        assert 'descriptive_stats' in results
        assert 'correlation_matrix' in results
        assert 'value1' in results['descriptive_stats']
        assert 'value2' in results['descriptive_stats']
    
    def test_inferential_tests(self, sample_csv_data):
        """Test inferential statistical tests."""
        analyzer = StatisticalAnalyzer()
        
        results = analyzer.perform_inferential_tests(sample_csv_data)
        
        assert 't_test' in results
        assert 'correlation_test' in results
        assert 'anova' in results
        assert 'p_value' in results['t_test']
        assert 'correlation_coefficient' in results['correlation_test']
    
    def test_regression_analysis(self, sample_csv_data):
        """Test regression analysis."""
        analyzer = StatisticalAnalyzer()
        
        results = analyzer.perform_regression_analysis(sample_csv_data)
        
        assert 'linear_regression' in results
        assert 'r_squared' in results['linear_regression']
        assert 'coefficient' in results['linear_regression']
        assert 'intercept' in results['linear_regression']

class TestFlaskRoutes:
    """Test cases for Flask routes."""
    
    def test_index_route(self, client):
        """Test the main dashboard route."""
        response = client.get('/')
        assert response.status_code == 200
        assert b'AutoStatIQ' in response.data
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get('/health')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
    
    def test_api_analyze_without_question(self, client):
        """Test API analyze endpoint without question."""
        response = client.post('/api/analyze', 
                              json={},
                              content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_upload_without_file(self, client):
        """Test upload endpoint without file."""
        response = client.post('/upload', data={})
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data

class TestReportGenerator:
    """Test cases for ReportGenerator class."""
    
    def test_report_creation(self):
        """Test DOCX report creation."""
        generator = ReportGenerator()
        
        sample_results = {
            'descriptive_stats': {'value1': {'mean': 3.5, 'std': 1.87}},
            'correlation_test': {'correlation_coefficient': 0.95, 'p_value': 0.001}
        }
        
        doc = generator.create_report(
            sample_results,
            "Sample interpretation",
            "Sample conclusion",
            "Test question",
            "Test dataset info"
        )
        
        assert doc is not None
        assert len(doc.paragraphs) > 0
    
    def test_descriptive_stats_table(self):
        """Test descriptive statistics table creation."""
        generator = ReportGenerator()
        
        desc_stats = {
            'value1': {'mean': 3.5, 'std': 1.87, 'count': 6},
            'value2': {'mean': 35.0, 'std': 18.7, 'count': 6}
        }
        
        # This should not raise an exception
        generator.add_descriptive_stats_table(desc_stats)

class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_allowed_file(self):
        """Test file extension validation."""
        from app import allowed_file
        
        assert allowed_file('test.csv') == True
        assert allowed_file('test.xlsx') == True
        assert allowed_file('test.docx') == True
        assert allowed_file('test.pdf') == True
        assert allowed_file('test.json') == True
        assert allowed_file('test.txt') == True
        assert allowed_file('test.exe') == False
        assert allowed_file('test') == False

if __name__ == '__main__':
    pytest.main(['-v'])
