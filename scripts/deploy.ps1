# AutoStatIQ PowerShell Deployment Script

param(
    [switch]$Pull,
    [switch]$Rebuild,
    [switch]$Logs
)

Write-Host "🚀 AutoStatIQ Deployment Script" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green

# Check if Docker is installed
try {
    $dockerVersion = docker --version
    Write-Host "✅ Docker found: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker is not installed. Please install Docker Desktop first." -ForegroundColor Red
    Write-Host "📥 Download from: https://www.docker.com/get-started" -ForegroundColor Yellow
    exit 1
}

# Check if docker-compose is installed
try {
    $composeVersion = docker-compose --version
    Write-Host "✅ Docker Compose found: $composeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker Compose is not installed." -ForegroundColor Red
    exit 1
}

# Create .env file if it doesn't exist
if (-not (Test-Path .env)) {
    Write-Host "📝 Creating .env file from template..." -ForegroundColor Yellow
    Copy-Item .env.example .env
    Write-Host "⚠️  IMPORTANT: Please update the .env file with your OpenAI API key!" -ForegroundColor Yellow
    Write-Host "   Edit .env and add: OPENAI_API_KEY=your_api_key_here" -ForegroundColor Yellow
    Read-Host "Press Enter after updating the .env file"
}

# Create necessary directories
Write-Host "📁 Creating necessary directories..." -ForegroundColor Blue
$directories = @("uploads", "results", "static\plots", "logs")
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

# Pull images if requested
if ($Pull) {
    Write-Host "📥 Pulling latest Docker images..." -ForegroundColor Blue
    docker-compose pull
}

# Build images
if ($Rebuild) {
    Write-Host "🏗️  Rebuilding Docker images..." -ForegroundColor Blue
    docker-compose build --no-cache
} else {
    Write-Host "🏗️  Building Docker images..." -ForegroundColor Blue
    docker-compose build
}

# Start services
Write-Host "🚀 Starting AutoStatIQ services..." -ForegroundColor Blue
docker-compose up -d

# Wait for application to start
Write-Host "⏳ Waiting for application to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 15

# Health check
Write-Host "🏥 Performing health check..." -ForegroundColor Blue
$maxRetries = 5
$retryCount = 0

while ($retryCount -lt $maxRetries) {
    try {
        $response = Invoke-WebRequest -Uri http://localhost:5000/health -UseBasicParsing -TimeoutSec 10
        if ($response.StatusCode -eq 200) {
            Write-Host "✅ AutoStatIQ is running successfully!" -ForegroundColor Green
            Write-Host ""
            Write-Host "🌐 Application URLs:" -ForegroundColor Cyan
            Write-Host "   Main App:     http://localhost:5000" -ForegroundColor White
            Write-Host "   API Docs:     http://localhost:5000/api-docs" -ForegroundColor White
            Write-Host "   Health Check: http://localhost:5000/health" -ForegroundColor White
            Write-Host ""
            Write-Host "📊 AutoStatIQ is ready for statistical analysis!" -ForegroundColor Green
            
            # Open browser
            Write-Host "🌐 Opening application in browser..." -ForegroundColor Blue
            Start-Process "http://localhost:5000"
            
            exit 0
        }
    } catch {
        Write-Host "⏳ Attempt $($retryCount + 1)/$maxRetries failed, retrying in 5 seconds..." -ForegroundColor Yellow
        Start-Sleep -Seconds 5
        $retryCount++
    }
}

Write-Host "❌ Health check failed after $maxRetries attempts." -ForegroundColor Red
Write-Host "📋 Checking logs for issues..." -ForegroundColor Yellow
docker-compose logs --tail=50 autostatiq

Write-Host ""
Write-Host "🔧 Troubleshooting steps:" -ForegroundColor Yellow
Write-Host "1. Check if port 5000 is available: netstat -an | findstr 5000" -ForegroundColor White
Write-Host "2. Verify .env file has correct OpenAI API key" -ForegroundColor White
Write-Host "3. Check Docker logs: docker-compose logs autostatiq" -ForegroundColor White
Write-Host "4. Restart services: docker-compose restart" -ForegroundColor White

if ($Logs) {
    Write-Host "📋 Showing detailed logs..." -ForegroundColor Blue
    docker-compose logs autostatiq
}

exit 1
