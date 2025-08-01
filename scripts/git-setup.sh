#!/bin/bash

# AutoStatIQ Git Repository Setup Script

echo "🐙 Setting up AutoStatIQ Git Repository"
echo "======================================="

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install Git first."
    exit 1
fi

# Initialize git repository
echo "📁 Initializing Git repository..."
git init

# Add all files to staging
echo "📝 Adding files to Git staging..."
git add .

# Create initial commit
echo "💾 Creating initial commit..."
git commit -m "Initial commit: AutoStatIQ v1.0.0

🚀 Features:
- Complete statistical analysis platform with 25+ methods
- Flask backend with advanced statistics engine
- Modern responsive frontend with Bootstrap 5
- Docker deployment ready with multi-stage builds
- Comprehensive REST API with client libraries
- Professional DOCX report generation
- GPT-4 integration for intelligent analysis
- Control charts and SPC analysis
- Machine learning capabilities (PCA, clustering)
- Multi-format file support (CSV, XLSX, DOCX, PDF, JSON, TXT)

🛠️ Technical Stack:
- Backend: Python Flask, NumPy, Pandas, SciPy, scikit-learn
- Frontend: HTML5, CSS3, JavaScript ES6+, Bootstrap 5
- Visualization: Matplotlib, Seaborn, Plotly.js
- AI: OpenAI GPT-4 API
- Deployment: Docker, Docker Compose, GitHub Actions

📚 Documentation:
- Complete API documentation with interactive examples
- Client libraries for Python, JavaScript, R
- Comprehensive deployment guides
- Docker and Git deployment strategies

👨‍💻 Author: Roman Chaudhary
🌐 Website: chaudharyroman.com.np
📧 Contact: contact@chaudharyroman.com.np"

echo "✅ Git repository initialized successfully!"
echo ""
echo "🔗 Next steps:"
echo "1. Create a new repository on GitHub: https://github.com/new"
echo "2. Repository name: rstat-auto (or autostatiq)"
echo "3. Add remote origin:"
echo "   git remote add origin https://github.com/romanch203/rstat-auto.git"
echo "4. Push to GitHub:"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "🐳 For Docker Hub deployment:"
echo "1. Create account at: https://hub.docker.com"
echo "2. Create repository: autostatiq"
echo "3. Build and push:"
echo "   docker build -t romanch203/autostatiq:latest ."
echo "   docker push romanch203/autostatiq:latest"
