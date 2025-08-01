# 🧪 AutoStatIQ Manual Testing Guide

## ✅ Everything is Fixed and Ready!

### 📁 **Test Data Files Created:**
- ✅ `sample_data.csv` - 20 employee records with 9 columns (Name, Age, Height, Weight, Salary, Department, Experience, Gender, City)
- ✅ `sample_data.json` - 5 employee records in JSON format
- ✅ All dependencies added to requirements.txt (factor-analyzer, lifelines)

### 🚀 **Manual Testing Steps:**

#### 1. **Start the Application**
Open PowerShell/Command Prompt in this directory and run:
```bash
python app.py
```

#### 2. **Open Browser**
Navigate to: http://localhost:5000

#### 3. **Test File Upload**
- Drag and drop `sample_data.csv` into the upload area
- Or click "Choose File" and select `sample_data.csv`

#### 4. **Test Analysis Questions**
Try these sample questions:

**Basic Statistics:**
- "Calculate descriptive statistics for all numeric columns"
- "What's the average salary by department?"
- "Show me the correlation between age and salary"

**Advanced Analysis:**
- "Perform regression analysis predicting salary from age and experience"
- "Create a scatter plot of salary vs experience"
- "Show me salary distribution by department"
- "Compare average salaries between Engineering and Marketing departments"

**Visualizations:**
- "Create a histogram of salaries"
- "Show a boxplot of salaries by department"
- "Generate a correlation heatmap"

#### 5. **Test API Endpoints**
Visit these URLs directly:
- http://localhost:5000/health (Should return "OK")
- http://localhost:5000/api-docs (API Documentation)
- http://localhost:5000/api/info (API Information)

#### 6. **Test Different File Formats**
- Try uploading `sample_data.json`
- Test with Excel files (.xlsx) if you have any

### 🔍 **What to Look For:**

#### ✅ **Expected Success Indicators:**
- Home page loads without errors
- File upload works smoothly
- Analysis questions return results
- Charts and visualizations display properly
- Download buttons work (DOCX reports)
- No console errors in browser developer tools

#### ❌ **Potential Issues to Check:**
- Import errors in terminal
- 500 Internal Server Error
- Charts not displaying
- File upload failures
- Missing dependencies warnings

### 📊 **Expected Analysis Results:**

**For sample_data.csv, you should see:**
- Average salary: ~$58,550
- Engineering department has highest average salary
- Positive correlation between experience and salary
- Gender and location distributions
- Age range: 26-36 years
- Experience range: 1-9 years

### 🎯 **Production Readiness Checklist:**

After successful local testing:
- [ ] All file uploads work
- [ ] Statistical calculations are accurate
- [ ] Visualizations render properly
- [ ] AI integration responds (if OpenAI key is set)
- [ ] Reports download successfully
- [ ] No Python errors in terminal
- [ ] API endpoints respond correctly

### 🚀 **Ready for Render Deployment!**

If all tests pass locally, the application is ready for Render cloud deployment. The missing dependencies have been fixed in requirements.txt:
- ✅ factor-analyzer==0.4.1
- ✅ lifelines==0.27.7

Your Render deployment should now work perfectly! 🎉

---

## 💡 **Troubleshooting:**

**If Flask won't start:**
1. Check if Python path is correct
2. Try: `python -m flask --app app run`
3. Ensure all dependencies are installed

**If imports fail:**
1. Install missing packages: `pip install -r requirements.txt`
2. Check Python version (should be 3.9+)

**If analysis fails:**
1. Check OpenAI API key in .env file
2. Verify file format is supported
3. Check file size (should be reasonable)
