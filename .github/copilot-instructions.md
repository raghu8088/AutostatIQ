<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# AutoStatIQ Development Instructions for GitHub Copilot

## Project Overview
AutoStatIQ is an AI-powered statistical analysis web application built with Python Flask backend and modern frontend technologies. The application accepts file uploads in various formats and natural language statistical questions, then provides comprehensive analysis reports using GPT-4 integration.

## Technology Stack
- **Backend**: Python Flask, Pandas, NumPy, SciPy, scikit-learn, OpenAI GPT-4
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5, Plotly.js, Dropzone.js
- **Data Processing**: python-docx, PyPDF2, openpyxl
- **Visualization**: Matplotlib, Seaborn, Plotly

## Code Style Guidelines
- Follow PEP 8 standards for Python code
- Use meaningful variable and function names
- Add comprehensive docstrings for all functions and classes
- Implement proper error handling with try-catch blocks
- Use type hints where appropriate
- Keep functions focused and modular

## Key Components

### Backend (`app.py`)
- **StatisticalAnalyzer**: Main class for data processing and analysis
- **ReportGenerator**: Handles DOCX report creation
- **Flask Routes**: RESTful API endpoints for upload, analysis, and download

### Frontend
- **templates/index.html**: Main dashboard interface
- **static/css/style.css**: Custom styling with CSS variables and animations
- **static/js/app.js**: JavaScript application logic with ES6 classes

## Development Patterns
- Use environment variables for sensitive configuration
- Implement proper file validation and security measures
- Create responsive designs that work on mobile and desktop
- Use modern JavaScript features (ES6+, async/await)
- Implement proper error handling and user feedback
- Follow RESTful API design principles

## Statistical Analysis Features
- Descriptive statistics (mean, median, mode, std dev)
- Correlation analysis (Pearson coefficients)
- Inferential tests (t-tests, ANOVA)
- Regression analysis (linear regression with R-squared)
- Data visualization (histograms, scatter plots, heatmaps)

## File Processing
- Support for CSV, XLSX, DOCX, PDF, JSON, TXT formats
- Automatic detection of structured vs. unstructured data
- Text extraction from documents
- Data validation and cleaning

## AI Integration
- OpenAI GPT-4 for natural language question processing
- Automated interpretation of statistical results
- Professional conclusion generation
- Context-aware recommendations

## Security Considerations
- Validate all file uploads
- Sanitize user inputs
- Use secure file handling practices
- Implement proper CORS configuration
- Protect API keys and sensitive data

## Testing Guidelines
- Test all statistical calculations for accuracy
- Validate file processing for all supported formats
- Test error handling scenarios
- Verify API endpoints functionality
- Ensure responsive design across devices

## Future Enhancement Areas
- Database integration for result storage
- User authentication and project management
- Additional statistical tests and methods
- Real-time collaboration features
- Integration with R for advanced analytics
- Export to additional formats (PDF, PowerPoint)

When suggesting code improvements or new features, prioritize:
1. Code maintainability and readability
2. User experience and accessibility
3. Statistical accuracy and reliability
4. Security and data protection
5. Performance optimization
