@echo off
echo Setting up AutoStatIQ Python environment...

REM Create virtual environment
python -m venv autostatiq_env

REM Activate virtual environment
call autostatiq_env\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install requirements
pip install -r requirements.txt

REM Install test requirements
pip install -r requirements-test.txt

REM Create .env file from example if it doesn't exist
if not exist .env (
    copy .env.example .env
    echo .env file created from example. Please edit it with your OpenAI API key.
)

echo Setup complete! 
echo.
echo Next steps:
echo 1. Edit .env file and add your OpenAI API key
echo 2. Run: autostatiq_env\Scripts\activate.bat
echo 3. Run: python app.py
echo.
pause
