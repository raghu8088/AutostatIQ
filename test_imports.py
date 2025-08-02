#!/usr/bin/env python3
"""
Test script to verify all critical imports work locally
"""

print("🧪 Testing AutoStatIQ Dependencies...")
print("=" * 50)

# Test core libraries
try:
    import pandas as pd
    import numpy as np
    import scipy
    print("✅ Core data libraries (pandas, numpy, scipy)")
except Exception as e:
    print(f"❌ Core data libraries failed: {e}")

# Test Flask
try:
    from flask import Flask
    print("✅ Flask web framework")
except Exception as e:
    print(f"❌ Flask failed: {e}")

# Test scientific libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    print("✅ Visualization libraries (matplotlib, seaborn, plotly)")
except Exception as e:
    print(f"❌ Visualization libraries failed: {e}")

# Test machine learning
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.cluster import KMeans
    print("✅ Scikit-learn machine learning")
except Exception as e:
    print(f"❌ Scikit-learn failed: {e}")

# Test advanced statistics
try:
    from factor_analyzer import FactorAnalyzer
    print("✅ Factor analyzer")
except Exception as e:
    print(f"❌ Factor analyzer failed: {e}")

# Test survival analysis
try:
    from lifelines import KaplanMeierFitter
    print("✅ Lifelines survival analysis")
except Exception as e:
    print(f"❌ Lifelines failed: {e}")

# Test OpenAI
try:
    import openai
    print("✅ OpenAI integration")
except Exception as e:
    print(f"❌ OpenAI failed: {e}")

# Test app import
try:
    from app import app
    print("✅ AutoStatIQ main application")
except Exception as e:
    print(f"❌ AutoStatIQ app failed: {e}")

print("=" * 50)
print("🎯 Import testing complete!")
