# AutoStatIQ Render Deployment Guide

## 🚀 Quick Deployment Steps

### 1. **Connect to Render**
1. Go to [render.com](https://render.com) and sign up/login
2. Click "New +" → "Web Service"
3. Connect your GitHub account
4. Select the `AutostatIQ` repository

### 2. **Configure Deployment**
- **Name**: `autostatiq`
- **Environment**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 300 app:app`
- **Instance Type**: `Free` (or upgrade as needed)

### 3. **Set Environment Variables**
Add these environment variables in Render dashboard:

```bash
OPENAI_API_KEY=your_openai_api_key_here
FLASK_ENV=production
SECRET_KEY=auto-generated-by-render
PYTHON_VERSION=3.9.16
```

### 4. **Deploy**
Click "Create Web Service" and Render will automatically:
- Pull your code from GitHub
- Install dependencies
- Start the application
- Provide you with a public URL

## 🔧 **Files Created for Render Deployment**

- ✅ `render.yaml` - Render service configuration
- ✅ `Procfile` - Process file for deployment
- ✅ `start.py` - Production start script
- ✅ Updated `app.py` - Production-ready Flask app
- ✅ Updated `requirements.txt` - All dependencies

## 🌐 **After Deployment**

Your AutoStatIQ will be live at:
`https://autostatiq-[random].onrender.com`

### Available Endpoints:
- **Main App**: `/`
- **API Documentation**: `/api-docs`
- **Health Check**: `/health`
- **API Info**: `/api/info`

## 🛠️ **Troubleshooting**

### If deployment fails:
1. Check Render logs for errors
2. Verify OPENAI_API_KEY is set correctly
3. Ensure all files are pushed to GitHub
4. Check Python version compatibility

### For support:
- Render Documentation: https://render.com/docs
- AutoStatIQ Issues: https://github.com/raghu8088/AutostatIQ/issues

## 🎯 **Production Features**

- ✅ **Auto-scaling**: Handles traffic spikes
- ✅ **HTTPS**: Secure connections by default
- ✅ **Health checks**: Automatic monitoring
- ✅ **Zero-downtime**: Smooth deployments
- ✅ **Custom domain**: Add your own domain (paid plans)

---

**AutoStatIQ** - Statistical Analysis in Sec, now cloud-ready! 🚀📊
