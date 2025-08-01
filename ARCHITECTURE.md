# AutoStatIQ - Project Architecture & Development Guide

## 🏗️ Architecture Overview

AutoStatIQ follows a modern full-stack architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────┐
│                Frontend Layer                    │
│  HTML5 + CSS3 + JavaScript (ES6+)              │
│  Bootstrap 5 + Plotly.js + Dropzone.js         │
└─────────────────────────────────────────────────┘
                         │
┌─────────────────────────────────────────────────┐
│                API Layer                        │
│  Flask RESTful API + Flask-CORS                 │
│  File Upload + Validation + Error Handling      │
└─────────────────────────────────────────────────┘
                         │
┌─────────────────────────────────────────────────┐
│              Business Logic Layer               │
│  StatisticalAnalyzer + ReportGenerator          │
│  Data Processing + Statistical Computing        │
└─────────────────────────────────────────────────┘
                         │
┌─────────────────────────────────────────────────┐
│               Integration Layer                 │
│  OpenAI GPT-4 API + Data Libraries             │
│  Pandas + NumPy + SciPy + scikit-learn         │
└─────────────────────────────────────────────────┘
```

## 🔧 Core Components

### 1. StatisticalAnalyzer Class
**Purpose**: Main engine for data analysis and statistical computations

**Key Methods**:
- `load_data()`: Multi-format file processing (CSV, XLSX, DOCX, PDF, JSON, TXT)
- `is_structured_data()`: Determines data type for appropriate analysis
- `perform_descriptive_analysis()`: Basic statistics and correlations
- `perform_inferential_tests()`: t-tests, ANOVA, correlation tests
- `perform_regression_analysis()`: Linear regression with R-squared
- `generate_plots()`: Interactive visualizations using Plotly
- `analyze_with_gpt()`: AI-powered question processing
- `get_gpt_interpretation()`: Statistical result interpretation
- `get_gpt_conclusion()`: Professional conclusions and recommendations

### 2. ReportGenerator Class
**Purpose**: Professional DOCX report creation

**Key Methods**:
- `create_report()`: Main report generation pipeline
- `add_results_tables()`: Statistical results formatting
- `add_descriptive_stats_table()`: Descriptive statistics tables
- `add_correlation_table()`: Correlation matrix visualization

### 3. Frontend JavaScript (AutoStatIQ Class)
**Purpose**: User interface management and API communication

**Key Methods**:
- `setupDropzone()`: Advanced file upload with drag & drop
- `handleFormSubmission()`: Form validation and submission
- `uploadFileForAnalysis()`: File-based analysis API calls
- `analyzeQuestionOnly()`: Text-based analysis API calls
- `displayResults()`: Dynamic result presentation
- `populateVisualizations()`: Interactive chart rendering
- `downloadReport()`: Report download management

## 📊 Statistical Analysis Pipeline

### Data Processing Flow
1. **File Upload** → File validation & secure storage
2. **Data Detection** → Structured vs. unstructured identification  
3. **Data Loading** → Format-specific parsing (CSV, Excel, PDF, etc.)
4. **Analysis Selection** → Automatic test selection based on data type
5. **Statistical Computing** → Multiple analysis types execution
6. **AI Integration** → GPT-4 interpretation and conclusions
7. **Report Generation** → Professional DOCX creation
8. **Visualization** → Interactive Plotly charts
9. **Download** → Secure file delivery

### Supported Statistical Tests
- **Descriptive Statistics**: Mean, median, mode, std dev, quartiles
- **Correlation Analysis**: Pearson correlation coefficients and p-values
- **t-Tests**: Independent samples t-tests for group comparisons
- **ANOVA**: Analysis of variance for multiple group comparisons
- **Linear Regression**: Simple regression with R-squared, coefficients
- **Frequency Analysis**: Categorical data distributions

## 🎨 Frontend Architecture

### CSS Architecture (BEM-inspired)
```css
/* Component-based styling */
.card { }                    /* Base component */
.card--primary { }           /* Component variation */
.card__header { }            /* Component element */
.card__header--gradient { }  /* Element variation */

/* Utility classes */
.text-gradient             
.shadow-custom
.fade-in-up
```

### JavaScript Architecture (ES6 Classes)
```javascript
class AutoStatIQ {
    constructor()           // Initialize application
    setupEventListeners()   // Bind UI events
    setupDropzone()         // Configure file upload
    handleFormSubmission()  // Process form data
    displayResults()        // Show analysis results
    downloadReport()        // Handle file downloads
}
```

## 🔌 API Design

### RESTful Endpoints
```
POST /upload                # File analysis
POST /api/analyze          # Question analysis  
GET  /download/<filename>  # Report download
GET  /health              # Health check
GET  /                    # Main dashboard
```

### Request/Response Patterns
```json
// File Upload Request
{
  "file": "multipart/form-data",
  "question": "optional statistical question"
}

// Analysis Response
{
  "success": true,
  "results": { /* statistical results */ },
  "interpretation": "AI interpretation text",
  "conclusion": "AI conclusion text", 
  "report_filename": "report_timestamp.docx",
  "plots": { /* plotly chart data */ }
}
```

## 🛡️ Security Implementation

### File Security
- **Extension Validation**: Whitelist approach for allowed file types
- **Size Limits**: 16MB maximum file size to prevent abuse
- **Secure Filenames**: `secure_filename()` for safe file handling
- **Temporary Storage**: Automatic cleanup of uploaded files
- **Path Validation**: Prevents directory traversal attacks

### Data Security  
- **Environment Variables**: Sensitive data (API keys) in .env files
- **Input Sanitization**: Validation for all user inputs
- **CORS Configuration**: Controlled cross-origin access
- **Session Security**: Flask secret key for session management

## 🚀 Performance Optimizations

### Backend Optimizations
- **Lazy Loading**: Data loaded only when needed
- **Vectorized Operations**: NumPy/Pandas for efficient computations
- **Memory Management**: Cleanup of large data structures
- **Background Processing**: Async support for long-running tasks

### Frontend Optimizations
- **Progressive Enhancement**: Works without JavaScript
- **Lazy Loading**: Charts rendered on-demand
- **Debounced Inputs**: Reduced API calls during typing
- **Caching**: Browser caching for static assets

## 🧪 Testing Strategy

### Unit Tests (`tests/test_app.py`)
- **StatisticalAnalyzer Tests**: Core analysis functions
- **ReportGenerator Tests**: DOCX generation
- **Flask Route Tests**: API endpoint validation
- **Utility Function Tests**: Helper function verification

### Test Categories
```python
class TestStatisticalAnalyzer:    # Core logic tests
class TestFlaskRoutes:           # API endpoint tests  
class TestReportGenerator:       # Report creation tests
class TestUtilityFunctions:      # Helper function tests
```

## 📦 Deployment Options

### Local Development
```bash
python app.py                    # Development server
```

### Docker Deployment  
```bash
docker build -t autostatiq .     # Build image
docker run -p 5000:5000 autostatiq  # Run container
```

### Docker Compose (with Nginx)
```bash
docker-compose up -d             # Production stack
```

### Cloud Deployment
- **Heroku**: `Procfile` for easy deployment
- **AWS**: Elastic Beanstalk or EC2 deployment
- **Google Cloud**: App Engine deployment
- **Azure**: App Service deployment

## 🔧 Configuration Management

### Environment Variables
```env
OPENAI_API_KEY=          # Required: OpenAI API access
SECRET_KEY=              # Required: Flask session security
FLASK_ENV=               # Optional: development/production
MAX_CONTENT_LENGTH=      # Optional: File size limit
UPLOAD_FOLDER=           # Optional: Upload directory
RESULTS_FOLDER=          # Optional: Results directory
```

### Configuration Files
- `.env`: Environment variables
- `requirements.txt`: Python dependencies
- `docker-compose.yml`: Container orchestration
- `.vscode/`: VS Code workspace configuration

## 🎯 Future Roadmap

### Phase 1: Core Enhancements
- [ ] Additional statistical tests (Chi-square, Mann-Whitney U)
- [ ] Support for time series analysis
- [ ] Advanced regression models (polynomial, logistic)
- [ ] Data cleaning and preprocessing options

### Phase 2: Platform Features  
- [ ] User authentication and project management
- [ ] Database integration for result storage
- [ ] Batch processing capabilities
- [ ] API rate limiting and quotas

### Phase 3: Advanced Analytics
- [ ] Machine learning model integration
- [ ] Real-time data streaming support
- [ ] Integration with R for advanced analytics
- [ ] Custom statistical test creation

### Phase 4: Enterprise Features
- [ ] Multi-user collaboration
- [ ] Advanced security features
- [ ] Custom branding options
- [ ] Enterprise SSO integration

## 📚 Developer Resources

### Key Libraries Documentation
- [Flask](https://flask.palletsprojects.com/): Web framework
- [Pandas](https://pandas.pydata.org/docs/): Data manipulation
- [SciPy](https://scipy.org/): Statistical computing
- [OpenAI](https://platform.openai.com/docs): AI integration
- [Plotly](https://plotly.com/javascript/): Data visualization

### Code Standards
- **Python**: PEP 8 compliance with Black formatting
- **JavaScript**: ES6+ features with proper error handling  
- **CSS**: BEM methodology with CSS custom properties
- **HTML**: Semantic markup with accessibility considerations

## 🤝 Contributing Guidelines

### Development Workflow
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for new functionality
4. Implement the feature
5. Ensure all tests pass
6. Submit pull request with detailed description

### Code Review Checklist
- [ ] Tests written and passing
- [ ] Documentation updated
- [ ] Code follows style guidelines
- [ ] Security considerations addressed
- [ ] Performance impact evaluated

---

This architecture ensures AutoStatIQ remains maintainable, scalable, and extensible while providing professional-grade statistical analysis capabilities.
