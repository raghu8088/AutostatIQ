# AutoStatIQ - Quick Start Guide

## Prerequisites
- Python 3.8+ installed
- OpenAI API key

## Quick Setup (Windows)
1. Run `setup.bat` to automatically create virtual environment and install dependencies
2. Edit `.env` file and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
3. Run the application:
   ```
   autostatiq_env\Scripts\activate.bat
   python app.py
   ```

## Manual Setup
1. Create virtual environment:
   ```
   python -m venv autostatiq_env
   autostatiq_env\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```
   copy .env.example .env
   # Edit .env file with your OpenAI API key
   ```

4. Run the application:
   ```
   python app.py
   ```

## Access the Application
Open your browser and go to: http://localhost:5000

## Features
- Upload CSV, XLSX, DOCX, PDF, JSON, or TXT files
- Ask statistical questions in natural language
- Get AI-powered analysis and interpretation
- Download professional DOCX reports
- Interactive visualizations

## Sample Data
Check the `sample_data/` folder for example files to test the application.

## Troubleshooting
- Make sure your OpenAI API key is valid and has credits
- Check that all dependencies are installed correctly
- Ensure Python 3.8+ is being used
- Verify the .env file is properly configured

## Contact
For issues or questions, visit: chaudharyroman.com.np
