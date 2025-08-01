"""
Advanced Statistical Analysis Module for AutoStatIQ
Comprehensive statistical methods for professional data analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats
from scipy.stats import chi2_contingency, shapiro, kstest, jarque_bera
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_curve, auc, accuracy_score, r2_score)
from sklearn.pipeline import Pipeline
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from factor_analyzer import FactorAnalyzer
from lifelines import KaplanMeierFitter, CoxPHFitter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import base64
import io
from typing import Dict, List, Tuple, Optional, Any

# Set matplotlib backend for server environments
plt.switch_backend('Agg')
warnings.filterwarnings('ignore')

class AdvancedStatisticalAnalyzer:
    """
    Comprehensive advanced statistical analysis engine
    """
    
    def __init__(self):
        self.results = {}
        self.plots = {}
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def perform_comprehensive_descriptive_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Enhanced descriptive statistics with skewness, kurtosis, and advanced metrics
        """
        results = {}
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            desc_stats = {}
            for col in numeric_cols:
                series = data[col].dropna()
                if len(series) > 0:
                    desc_stats[col] = {
                        'count': len(series),
                        'mean': float(series.mean()),
                        'median': float(series.median()),
                        'mode': float(series.mode().iloc[0]) if len(series.mode()) > 0 else np.nan,
                        'std': float(series.std()),
                        'variance': float(series.var()),
                        'min': float(series.min()),
                        'max': float(series.max()),
                        'range': float(series.max() - series.min()),
                        'q25': float(series.quantile(0.25)),
                        'q75': float(series.quantile(0.75)),
                        'iqr': float(series.quantile(0.75) - series.quantile(0.25)),
                        'skewness': float(series.skew()),
                        'kurtosis': float(series.kurtosis()),
                        'cv': float(series.std() / series.mean()) if series.mean() != 0 else np.nan
                    }
            
            results['enhanced_descriptive_stats'] = desc_stats
        
        return results
    
    def perform_frequency_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive frequency analysis for categorical variables with visualizations
        """
        results = {}
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) > 0:
            freq_analysis = {}
            
            for col in categorical_cols:
                series = data[col].dropna()
                if len(series) > 0:
                    freq_counts = series.value_counts()
                    freq_props = series.value_counts(normalize=True)
                    
                    freq_analysis[col] = {
                        'frequencies': freq_counts.to_dict(),
                        'proportions': freq_props.to_dict(),
                        'unique_count': len(freq_counts),
                        'most_frequent': freq_counts.index[0],
                        'least_frequent': freq_counts.index[-1]
                    }
                    
                    # Create bar plot
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Bar plot
                    freq_counts.plot(kind='bar', ax=ax1, color='skyblue')
                    ax1.set_title(f'Frequency Distribution - {col}')
                    ax1.set_xlabel(col)
                    ax1.set_ylabel('Frequency')
                    ax1.tick_params(axis='x', rotation=45)
                    
                    # Pie chart
                    if len(freq_counts) <= 10:  # Only show pie chart for reasonable number of categories
                        ax2.pie(freq_counts.values, labels=freq_counts.index, autopct='%1.1f%%')
                        ax2.set_title(f'Proportion Distribution - {col}')
                    else:
                        # Show top 10 categories for pie chart
                        top_10 = freq_counts.head(10)
                        others = freq_counts.iloc[10:].sum()
                        if others > 0:
                            plot_data = top_10.copy()
                            plot_data['Others'] = others
                        else:
                            plot_data = top_10
                        ax2.pie(plot_data.values, labels=plot_data.index, autopct='%1.1f%%')
                        ax2.set_title(f'Top Categories Distribution - {col}')
                    
                    plt.tight_layout()
                    
                    # Save plot as base64
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                    buffer.seek(0)
                    plot_base64 = base64.b64encode(buffer.getvalue()).decode()
                    buffer.close()
                    plt.close()
                    
                    freq_analysis[col]['plot'] = plot_base64
            
            results['frequency_analysis'] = freq_analysis
        
        return results
    
    def perform_cross_tabulation_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Cross-tabulation and Chi-square tests for categorical variable independence
        """
        results = {}
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) >= 2:
            cross_tab_results = {}
            
            # Test all pairs of categorical variables
            for i, col1 in enumerate(categorical_cols):
                for col2 in categorical_cols[i+1:]:
                    pair_key = f"{col1}_vs_{col2}"
                    
                    # Create contingency table
                    contingency_table = pd.crosstab(data[col1], data[col2])
                    
                    # Perform Chi-square test
                    try:
                        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                        
                        cross_tab_results[pair_key] = {
                            'contingency_table': contingency_table.to_dict(),
                            'chi2_statistic': float(chi2),
                            'p_value': float(p_value),
                            'degrees_of_freedom': int(dof),
                            'expected_frequencies': pd.DataFrame(expected, 
                                                               index=contingency_table.index,
                                                               columns=contingency_table.columns).to_dict(),
                            'significant': p_value < 0.05,
                            'interpretation': self._interpret_chi_square(p_value)
                        }
                        
                        # Create heatmap visualization
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues', ax=ax)
                        ax.set_title(f'Cross-tabulation: {col1} vs {col2}')
                        
                        # Save plot
                        buffer = io.BytesIO()
                        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                        buffer.seek(0)
                        plot_base64 = base64.b64encode(buffer.getvalue()).decode()
                        buffer.close()
                        plt.close()
                        
                        cross_tab_results[pair_key]['heatmap'] = plot_base64
                        
                    except Exception as e:
                        print(f"Error in chi-square test for {pair_key}: {str(e)}")
                        continue
            
            results['cross_tabulation'] = cross_tab_results
        
        return results
    
    def perform_advanced_correlation_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Advanced correlation analysis with multiple correlation types and visualizations
        """
        results = {}
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) >= 2:
            numeric_data = data[numeric_cols].dropna()
            
            if len(numeric_data) > 0:
                correlation_results = {}
                
                # Calculate different types of correlations
                pearson_corr = numeric_data.corr(method='pearson')
                spearman_corr = numeric_data.corr(method='spearman')
                kendall_corr = numeric_data.corr(method='kendall')
                
                correlation_results = {
                    'pearson': pearson_corr.to_dict(),
                    'spearman': spearman_corr.to_dict(),
                    'kendall': kendall_corr.to_dict()
                }
                
                # Create correlation heatmaps
                fig, axes = plt.subplots(2, 2, figsize=(20, 16))
                
                # Pearson correlation heatmap
                sns.heatmap(pearson_corr, annot=True, cmap='RdBu_r', center=0, 
                           square=True, ax=axes[0,0])
                axes[0,0].set_title('Pearson Correlation Matrix')
                
                # Spearman correlation heatmap
                sns.heatmap(spearman_corr, annot=True, cmap='RdBu_r', center=0, 
                           square=True, ax=axes[0,1])
                axes[0,1].set_title('Spearman Correlation Matrix')
                
                # Kendall correlation heatmap
                sns.heatmap(kendall_corr, annot=True, cmap='RdBu_r', center=0, 
                           square=True, ax=axes[1,0])
                axes[1,0].set_title('Kendall Correlation Matrix')
                
                # Scatterplot matrix for first 5 variables (if more than 5)
                if len(numeric_cols) > 5:
                    plot_cols = numeric_cols[:5]
                else:
                    plot_cols = numeric_cols
                    
                # Create pairplot
                axes[1,1].remove()
                plt.tight_layout()
                
                # Save correlation plots
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                buffer.seek(0)
                plot_base64 = base64.b64encode(buffer.getvalue()).decode()
                buffer.close()
                plt.close()
                
                correlation_results['correlation_heatmaps'] = plot_base64
                
                # Create separate pairplot
                if len(plot_cols) <= 5:
                    pair_plot = sns.pairplot(numeric_data[plot_cols])
                    pair_plot.fig.suptitle('Scatterplot Matrix', y=1.02)
                    
                    buffer = io.BytesIO()
                    pair_plot.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                    buffer.seek(0)
                    pairplot_base64 = base64.b64encode(buffer.getvalue()).decode()
                    buffer.close()
                    plt.close()
                    
                    correlation_results['pairplot'] = pairplot_base64
                
                # Calculate significance for Pearson correlations
                pearson_pvalues = {}
                for col1 in numeric_cols:
                    pearson_pvalues[col1] = {}
                    for col2 in numeric_cols:
                        if col1 != col2:
                            try:
                                corr, p_val = stats.pearsonr(numeric_data[col1], numeric_data[col2])
                                pearson_pvalues[col1][col2] = float(p_val)
                            except:
                                pearson_pvalues[col1][col2] = np.nan
                        else:
                            pearson_pvalues[col1][col2] = 0.0
                
                correlation_results['pearson_pvalues'] = pearson_pvalues
                results['advanced_correlation'] = correlation_results
        
        return results
    
    def perform_advanced_regression_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Advanced regression analysis with diagnostics and visualizations
        """
        results = {}
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) >= 2:
            regression_results = {}
            
            # Try different regression models
            for target_col in numeric_cols:
                predictor_cols = [col for col in numeric_cols if col != target_col]
                
                if len(predictor_cols) > 0:
                    # Prepare data
                    reg_data = data[list(predictor_cols) + [target_col]].dropna()
                    
                    if len(reg_data) > 10:  # Minimum sample size
                        X = reg_data[predictor_cols]
                        y = reg_data[target_col]
                        
                        try:
                            # Simple linear regression (first predictor)
                            simple_reg = self._perform_simple_regression(X.iloc[:, 0], y, 
                                                                       predictor_cols[0], target_col)
                            
                            # Multiple regression (if more than one predictor)
                            if len(predictor_cols) > 1:
                                multiple_reg = self._perform_multiple_regression(X, y, 
                                                                               predictor_cols, target_col)
                                regression_results[f"{target_col}_multiple"] = multiple_reg
                            
                            regression_results[f"{target_col}_simple"] = simple_reg
                            
                        except Exception as e:
                            print(f"Error in regression for {target_col}: {str(e)}")
                            continue
            
            results['advanced_regression'] = regression_results
        
        return results
    
    def perform_logistic_regression_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Logistic regression for binary classification with ROC curves
        """
        results = {}
        
        # Look for binary variables
        binary_cols = []
        for col in data.columns:
            unique_vals = data[col].dropna().unique()
            if len(unique_vals) == 2:
                binary_cols.append(col)
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(binary_cols) > 0 and len(numeric_cols) > 0:
            logistic_results = {}
            
            for target_col in binary_cols:
                predictor_cols = [col for col in numeric_cols if col != target_col]
                
                if len(predictor_cols) > 0:
                    # Prepare data
                    log_data = data[list(predictor_cols) + [target_col]].dropna()
                    
                    if len(log_data) > 20:  # Minimum sample size
                        X = log_data[predictor_cols]
                        y = log_data[target_col]
                        
                        # Encode target if necessary
                        if y.dtype == 'object':
                            le = LabelEncoder()
                            y = le.fit_transform(y)
                        
                        try:
                            # Split data
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.3, random_state=42
                            )
                            
                            # Fit logistic regression
                            log_reg = LogisticRegression(random_state=42)
                            log_reg.fit(X_train, y_train)
                            
                            # Predictions
                            y_pred = log_reg.predict(X_test)
                            y_pred_proba = log_reg.predict_proba(X_test)[:, 1]
                            
                            # Calculate metrics
                            accuracy = accuracy_score(y_test, y_pred)
                            
                            # ROC curve
                            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
                            roc_auc = auc(fpr, tpr)
                            
                            # Create ROC plot
                            fig, ax = plt.subplots(figsize=(8, 6))
                            ax.plot(fpr, tpr, color='darkorange', lw=2, 
                                   label=f'ROC curve (AUC = {roc_auc:.2f})')
                            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                            ax.set_xlim([0.0, 1.0])
                            ax.set_ylim([0.0, 1.05])
                            ax.set_xlabel('False Positive Rate')
                            ax.set_ylabel('True Positive Rate')
                            ax.set_title(f'ROC Curve - {target_col}')
                            ax.legend(loc="lower right")
                            
                            # Save ROC plot
                            buffer = io.BytesIO()
                            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                            buffer.seek(0)
                            roc_plot = base64.b64encode(buffer.getvalue()).decode()
                            buffer.close()
                            plt.close()
                            
                            # Odds ratios
                            odds_ratios = np.exp(log_reg.coef_[0])
                            
                            logistic_results[target_col] = {
                                'coefficients': dict(zip(predictor_cols, log_reg.coef_[0])),
                                'odds_ratios': dict(zip(predictor_cols, odds_ratios)),
                                'intercept': float(log_reg.intercept_[0]),
                                'accuracy': float(accuracy),
                                'auc': float(roc_auc),
                                'roc_plot': roc_plot,
                                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
                            }
                            
                        except Exception as e:
                            print(f"Error in logistic regression for {target_col}: {str(e)}")
                            continue
            
            results['logistic_regression'] = logistic_results
        
        return results
    
    def perform_comprehensive_hypothesis_testing(self, data: pd.DataFrame, user_question: str = "") -> Dict[str, Any]:
        """
        Comprehensive hypothesis testing based on data characteristics and user query
        """
        results = {}
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        hypothesis_results = {}
        
        # One-sample t-tests (test if means differ from hypothetical values)
        if len(numeric_cols) > 0:
            one_sample_tests = {}
            for col in numeric_cols:
                series = data[col].dropna()
                if len(series) > 3:
                    # Test against population mean (using overall mean as reference)
                    pop_mean = series.mean()
                    sample = series.sample(min(30, len(series)))
                    
                    if len(sample) > 1:
                        t_stat, p_val = stats.ttest_1samp(sample, pop_mean)
                        
                        one_sample_tests[col] = {
                            'statistic': float(t_stat),
                            'p_value': float(p_val),
                            'sample_mean': float(sample.mean()),
                            'population_mean': float(pop_mean),
                            'significant': p_val < 0.05,
                            'interpretation': self._interpret_p_value(p_val)
                        }
            
            hypothesis_results['one_sample_tests'] = one_sample_tests
        
        # Two-sample t-tests between groups
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            two_sample_tests = {}
            
            for cat_col in categorical_cols:
                unique_groups = data[cat_col].dropna().unique()
                if len(unique_groups) == 2:  # Binary categorical variable
                    for num_col in numeric_cols:
                        test_data = data[[cat_col, num_col]].dropna()
                        
                        if len(test_data) > 10:
                            group1 = test_data[test_data[cat_col] == unique_groups[0]][num_col]
                            group2 = test_data[test_data[cat_col] == unique_groups[1]][num_col]
                            
                            if len(group1) > 1 and len(group2) > 1:
                                # Independent t-test
                                t_stat, p_val = stats.ttest_ind(group1, group2)
                                
                                # Mann-Whitney U test (non-parametric alternative)
                                u_stat, u_p_val = stats.mannwhitneyu(group1, group2, 
                                                                    alternative='two-sided')
                                
                                test_key = f"{num_col}_by_{cat_col}"
                                two_sample_tests[test_key] = {
                                    't_test': {
                                        'statistic': float(t_stat),
                                        'p_value': float(p_val),
                                        'significant': p_val < 0.05,
                                        'interpretation': self._interpret_p_value(p_val)
                                    },
                                    'mann_whitney': {
                                        'statistic': float(u_stat),
                                        'p_value': float(u_p_val),
                                        'significant': u_p_val < 0.05,
                                        'interpretation': self._interpret_p_value(u_p_val)
                                    },
                                    'group1_mean': float(group1.mean()),
                                    'group2_mean': float(group2.mean()),
                                    'group1_name': str(unique_groups[0]),
                                    'group2_name': str(unique_groups[1])
                                }
            
            hypothesis_results['two_sample_tests'] = two_sample_tests
        
        results['hypothesis_testing'] = hypothesis_results
        return results
    
    def perform_advanced_anova(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Advanced ANOVA with post-hoc testing and visualizations
        """
        results = {}
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            anova_results = {}
            
            for cat_col in categorical_cols:
                unique_groups = data[cat_col].dropna().unique()
                if len(unique_groups) >= 3:  # Need at least 3 groups for ANOVA
                    
                    for num_col in numeric_cols:
                        test_data = data[[cat_col, num_col]].dropna()
                        
                        if len(test_data) > 15:  # Minimum sample size
                            groups = [test_data[test_data[cat_col] == group][num_col].values 
                                    for group in unique_groups]
                            
                            # Remove empty groups
                            groups = [group for group in groups if len(group) > 0]
                            
                            if len(groups) >= 3:
                                try:
                                    # One-way ANOVA
                                    f_stat, p_val = stats.f_oneway(*groups)
                                    
                                    # Kruskal-Wallis test (non-parametric alternative)
                                    kw_stat, kw_p_val = stats.kruskal(*groups)
                                    
                                    test_key = f"{num_col}_by_{cat_col}"
                                    
                                    anova_result = {
                                        'anova': {
                                            'f_statistic': float(f_stat),
                                            'p_value': float(p_val),
                                            'significant': p_val < 0.05,
                                            'interpretation': self._interpret_p_value(p_val)
                                        },
                                        'kruskal_wallis': {
                                            'statistic': float(kw_stat),
                                            'p_value': float(kw_p_val),
                                            'significant': kw_p_val < 0.05,
                                            'interpretation': self._interpret_p_value(kw_p_val)
                                        },
                                        'group_statistics': {}
                                    }
                                    
                                    # Group statistics
                                    for i, group_name in enumerate(unique_groups):
                                        if i < len(groups):
                                            group_data = groups[i]
                                            anova_result['group_statistics'][str(group_name)] = {
                                                'mean': float(np.mean(group_data)),
                                                'std': float(np.std(group_data)),
                                                'count': len(group_data)
                                            }
                                    
                                    # Post-hoc test if ANOVA is significant
                                    if p_val < 0.05:
                                        try:
                                            tukey_result = pairwise_tukeyhsd(
                                                test_data[num_col], test_data[cat_col]
                                            )
                                            anova_result['tukey_hsd'] = {
                                                'summary': str(tukey_result),
                                                'pairwise_significant': []
                                            }
                                            
                                            # Extract significant pairs
                                            for i in range(len(tukey_result.pvalues)):
                                                if tukey_result.pvalues[i] < 0.05:
                                                    anova_result['tukey_hsd']['pairwise_significant'].append({
                                                        'group1': tukey_result.groupsunique[tukey_result.group1[i]],
                                                        'group2': tukey_result.groupsunique[tukey_result.group2[i]],
                                                        'p_value': float(tukey_result.pvalues[i])
                                                    })
                                        except Exception as e:
                                            print(f"Error in Tukey HSD for {test_key}: {str(e)}")
                                    
                                    # Create box plot
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    test_data.boxplot(column=num_col, by=cat_col, ax=ax)
                                    ax.set_title(f'Box Plot: {num_col} by {cat_col}')
                                    ax.set_xlabel(cat_col)
                                    ax.set_ylabel(num_col)
                                    plt.suptitle('')  # Remove default title
                                    
                                    # Save plot
                                    buffer = io.BytesIO()
                                    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                                    buffer.seek(0)
                                    plot_base64 = base64.b64encode(buffer.getvalue()).decode()
                                    buffer.close()
                                    plt.close()
                                    
                                    anova_result['boxplot'] = plot_base64
                                    anova_results[test_key] = anova_result
                                    
                                except Exception as e:
                                    print(f"Error in ANOVA for {test_key}: {str(e)}")
                                    continue
            
            results['advanced_anova'] = anova_results
        
        return results
    
    def perform_normality_tests(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive normality testing with multiple methods
        """
        results = {}
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            normality_results = {}
            
            for col in numeric_cols:
                series = data[col].dropna()
                if len(series) > 3:
                    col_results = {}
                    
                    # Shapiro-Wilk test
                    if len(series) <= 5000:  # Shapiro-Wilk has sample size limit
                        try:
                            sw_stat, sw_p = shapiro(series)
                            col_results['shapiro_wilk'] = {
                                'statistic': float(sw_stat),
                                'p_value': float(sw_p),
                                'normal': sw_p > 0.05,
                                'interpretation': 'Normal distribution' if sw_p > 0.05 else 'Not normal distribution'
                            }
                        except Exception as e:
                            print(f"Shapiro-Wilk error for {col}: {str(e)}")
                    
                    # Kolmogorov-Smirnov test
                    try:
                        ks_stat, ks_p = kstest(series, 'norm', args=(series.mean(), series.std()))
                        col_results['kolmogorov_smirnov'] = {
                            'statistic': float(ks_stat),
                            'p_value': float(ks_p),
                            'normal': ks_p > 0.05,
                            'interpretation': 'Normal distribution' if ks_p > 0.05 else 'Not normal distribution'
                        }
                    except Exception as e:
                        print(f"KS test error for {col}: {str(e)}")
                    
                    # Jarque-Bera test
                    try:
                        jb_stat, jb_p = jarque_bera(series)
                        col_results['jarque_bera'] = {
                            'statistic': float(jb_stat),
                            'p_value': float(jb_p),
                            'normal': jb_p > 0.05,
                            'interpretation': 'Normal distribution' if jb_p > 0.05 else 'Not normal distribution'
                        }
                    except Exception as e:
                        print(f"Jarque-Bera error for {col}: {str(e)}")
                    
                    # Create Q-Q plot and histogram
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # Q-Q plot
                    stats.probplot(series, dist="norm", plot=ax1)
                    ax1.set_title(f'Q-Q Plot - {col}')
                    
                    # Histogram with normal curve overlay
                    ax2.hist(series, bins=30, density=True, alpha=0.7, color='skyblue')
                    x = np.linspace(series.min(), series.max(), 100)
                    normal_curve = stats.norm.pdf(x, series.mean(), series.std())
                    ax2.plot(x, normal_curve, 'r-', linewidth=2, label='Normal Distribution')
                    ax2.set_title(f'Histogram with Normal Curve - {col}')
                    ax2.legend()
                    
                    plt.tight_layout()
                    
                    # Save plot
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                    buffer.seek(0)
                    plot_base64 = base64.b64encode(buffer.getvalue()).decode()
                    buffer.close()
                    plt.close()
                    
                    col_results['normality_plots'] = plot_base64
                    normality_results[col] = col_results
            
            results['normality_tests'] = normality_results
        
        return results
    
    def perform_outlier_detection(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive outlier detection using multiple methods
        """
        results = {}
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            outlier_results = {}
            
            for col in numeric_cols:
                series = data[col].dropna()
                if len(series) > 3:
                    col_results = {}
                    
                    # Z-score method
                    z_scores = np.abs(stats.zscore(series))
                    z_outliers = series[z_scores > 3]
                    
                    col_results['z_score'] = {
                        'outlier_count': len(z_outliers),
                        'outlier_indices': z_outliers.index.tolist(),
                        'outlier_values': z_outliers.tolist(),
                        'threshold': 3.0
                    }
                    
                    # IQR method
                    Q1 = series.quantile(0.25)
                    Q3 = series.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    iqr_outliers = series[(series < lower_bound) | (series > upper_bound)]
                    
                    col_results['iqr'] = {
                        'outlier_count': len(iqr_outliers),
                        'outlier_indices': iqr_outliers.index.tolist(),
                        'outlier_values': iqr_outliers.tolist(),
                        'lower_bound': float(lower_bound),
                        'upper_bound': float(upper_bound)
                    }
                    
                    # Create box plot with outliers highlighted
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # Box plot
                    box_plot = ax1.boxplot(series, patch_artist=True)
                    box_plot['boxes'][0].set_facecolor('lightblue')
                    ax1.set_title(f'Box Plot with Outliers - {col}')
                    ax1.set_ylabel(col)
                    
                    # Scatter plot with outliers highlighted
                    ax2.scatter(range(len(series)), series, alpha=0.6, label='Normal')
                    if len(iqr_outliers) > 0:
                        outlier_indices = [list(series.index).index(idx) for idx in iqr_outliers.index]
                        ax2.scatter(outlier_indices, iqr_outliers.values, 
                                  color='red', s=50, label='IQR Outliers')
                    ax2.set_title(f'Data Points with Outliers - {col}')
                    ax2.set_xlabel('Index')
                    ax2.set_ylabel(col)
                    ax2.legend()
                    
                    plt.tight_layout()
                    
                    # Save plot
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                    buffer.seek(0)
                    plot_base64 = base64.b64encode(buffer.getvalue()).decode()
                    buffer.close()
                    plt.close()
                    
                    col_results['outlier_plots'] = plot_base64
                    outlier_results[col] = col_results
            
            results['outlier_detection'] = outlier_results
        
        return results
    
    def perform_missing_data_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive missing data analysis and handling strategies
        """
        results = {}
        
        # Missing data summary
        missing_summary = {}
        for col in data.columns:
            missing_count = data[col].isnull().sum()
            total_count = len(data)
            missing_pct = (missing_count / total_count) * 100
            
            missing_summary[col] = {
                'missing_count': int(missing_count),
                'total_count': int(total_count),
                'missing_percentage': float(missing_pct),
                'complete_count': int(total_count - missing_count)
            }
        
        # Missing data pattern analysis
        missing_pattern = data.isnull().sum().to_dict()
        
        # Create missing data visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Missing data bar chart
        missing_counts = [missing_summary[col]['missing_count'] for col in data.columns]
        ax1.bar(data.columns, missing_counts, color='lightcoral')
        ax1.set_title('Missing Data Count by Column')
        ax1.set_xlabel('Columns')
        ax1.set_ylabel('Missing Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Missing data heatmap
        sns.heatmap(data.isnull(), cbar=True, yticklabels=False, cmap='viridis', ax=ax2)
        ax2.set_title('Missing Data Pattern Heatmap')
        
        plt.tight_layout()
        
        # Save plot
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        plt.close()
        
        results['missing_data_analysis'] = {
            'summary': missing_summary,
            'pattern': missing_pattern,
            'total_rows': len(data),
            'complete_cases': len(data.dropna()),
            'visualization': plot_base64
        }
        
        return results
    
    def perform_principal_component_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Principal Component Analysis for dimensionality reduction
        """
        results = {}
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) >= 3:  # Need at least 3 variables for meaningful PCA
            pca_data = data[numeric_cols].dropna()
            
            if len(pca_data) > 10:
                # Standardize the data
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(pca_data)
                
                # Perform PCA
                pca = PCA()
                pca_result = pca.fit_transform(scaled_data)
                
                # Calculate explained variance
                explained_variance = pca.explained_variance_ratio_
                cumulative_variance = np.cumsum(explained_variance)
                
                # Determine optimal number of components (Kaiser criterion: eigenvalue > 1)
                eigenvalues = pca.explained_variance_
                optimal_components = np.sum(eigenvalues > 1)
                
                pca_results = {
                    'explained_variance_ratio': explained_variance.tolist(),
                    'cumulative_variance': cumulative_variance.tolist(),
                    'eigenvalues': eigenvalues.tolist(),
                    'optimal_components': int(optimal_components),
                    'total_variance_explained': float(cumulative_variance[optimal_components-1] if optimal_components > 0 else 0)
                }
                
                # Component loadings
                loadings = pca.components_[:optimal_components if optimal_components > 0 else 3]
                loading_df = pd.DataFrame(
                    loadings.T,
                    columns=[f'PC{i+1}' for i in range(len(loadings))],
                    index=numeric_cols
                )
                pca_results['component_loadings'] = loading_df.to_dict()
                
                # Create visualizations
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
                
                # Scree plot
                ax1.plot(range(1, len(explained_variance) + 1), explained_variance, 'bo-')
                ax1.axhline(y=1/len(numeric_cols), color='r', linestyle='--', 
                           label='Random Expectation')
                ax1.set_xlabel('Principal Component')
                ax1.set_ylabel('Explained Variance Ratio')
                ax1.set_title('Scree Plot')
                ax1.legend()
                
                # Cumulative variance plot
                ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'ro-')
                ax2.axhline(y=0.8, color='g', linestyle='--', label='80% Variance')
                ax2.set_xlabel('Number of Components')
                ax2.set_ylabel('Cumulative Explained Variance')
                ax2.set_title('Cumulative Explained Variance')
                ax2.legend()
                
                # Biplot (if we have at least 2 components)
                if len(pca_result[0]) >= 2:
                    ax3.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6)
                    ax3.set_xlabel(f'PC1 ({explained_variance[0]:.2%} variance)')
                    ax3.set_ylabel(f'PC2 ({explained_variance[1]:.2%} variance)')
                    ax3.set_title('PCA Biplot')
                    
                    # Add loading vectors
                    for i, var in enumerate(numeric_cols):
                        ax3.arrow(0, 0, loadings[0, i]*3, loadings[1, i]*3, 
                                head_width=0.1, head_length=0.1, fc='red', ec='red')
                        ax3.text(loadings[0, i]*3.2, loadings[1, i]*3.2, var, 
                               fontsize=8, ha='center', va='center')
                
                # Component loadings heatmap
                if len(loading_df.columns) > 1:
                    sns.heatmap(loading_df.T, annot=True, cmap='RdBu_r', center=0, ax=ax4)
                    ax4.set_title('Component Loadings Heatmap')
                
                plt.tight_layout()
                
                # Save plot
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                buffer.seek(0)
                plot_base64 = base64.b64encode(buffer.getvalue()).decode()
                buffer.close()
                plt.close()
                
                pca_results['pca_plots'] = plot_base64
                results['principal_component_analysis'] = pca_results
        
        return results
    
    def perform_clustering_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive clustering analysis with multiple algorithms
        """
        results = {}
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) >= 2:
            cluster_data = data[numeric_cols].dropna()
            
            if len(cluster_data) > 10:
                # Standardize data
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(cluster_data)
                
                clustering_results = {}
                
                # K-means clustering
                # Determine optimal number of clusters using elbow method
                inertias = []
                k_range = range(2, min(11, len(cluster_data)//2))
                
                for k in k_range:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans.fit(scaled_data)
                    inertias.append(kmeans.inertia_)
                
                # Find elbow point (simple method)
                optimal_k = 3  # Default
                if len(inertias) > 2:
                    # Use elbow method approximation
                    diffs = np.diff(inertias)
                    diffs2 = np.diff(diffs)
                    if len(diffs2) > 0:
                        optimal_k = np.argmax(diffs2) + 3  # +3 because we start from k=2 and take second diff
                
                # Perform K-means with optimal k
                kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                kmeans_labels = kmeans_optimal.fit_predict(scaled_data)
                
                clustering_results['kmeans'] = {
                    'optimal_clusters': int(optimal_k),
                    'inertia': float(kmeans_optimal.inertia_),
                    'cluster_centers': kmeans_optimal.cluster_centers_.tolist(),
                    'labels': kmeans_labels.tolist()
                }
                
                # DBSCAN clustering
                try:
                    dbscan = DBSCAN(eps=0.5, min_samples=5)
                    dbscan_labels = dbscan.fit_predict(scaled_data)
                    n_clusters_db = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
                    n_noise = list(dbscan_labels).count(-1)
                    
                    clustering_results['dbscan'] = {
                        'n_clusters': int(n_clusters_db),
                        'n_noise_points': int(n_noise),
                        'labels': dbscan_labels.tolist(),
                        'eps': 0.5,
                        'min_samples': 5
                    }
                except Exception as e:
                    print(f"DBSCAN error: {str(e)}")
                
                # Create visualizations
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                
                # Elbow plot
                axes[0, 0].plot(k_range, inertias, 'bo-')
                axes[0, 0].axvline(x=optimal_k, color='r', linestyle='--', 
                                  label=f'Optimal k={optimal_k}')
                axes[0, 0].set_xlabel('Number of Clusters (k)')
                axes[0, 0].set_ylabel('Inertia')
                axes[0, 0].set_title('Elbow Method for Optimal k')
                axes[0, 0].legend()
                
                # K-means clustering scatter plot (first two dimensions)
                if scaled_data.shape[1] >= 2:
                    scatter = axes[0, 1].scatter(scaled_data[:, 0], scaled_data[:, 1], 
                                               c=kmeans_labels, cmap='viridis', alpha=0.6)
                    axes[0, 1].scatter(kmeans_optimal.cluster_centers_[:, 0], 
                                      kmeans_optimal.cluster_centers_[:, 1],
                                      c='red', marker='x', s=200, linewidths=3, 
                                      label='Centroids')
                    axes[0, 1].set_xlabel(f'{numeric_cols[0]} (standardized)')
                    axes[0, 1].set_ylabel(f'{numeric_cols[1]} (standardized)')
                    axes[0, 1].set_title('K-means Clustering')
                    axes[0, 1].legend()
                    plt.colorbar(scatter, ax=axes[0, 1])
                
                # DBSCAN clustering scatter plot
                if 'dbscan' in clustering_results and scaled_data.shape[1] >= 2:
                    scatter_db = axes[1, 0].scatter(scaled_data[:, 0], scaled_data[:, 1], 
                                                   c=dbscan_labels, cmap='viridis', alpha=0.6)
                    axes[1, 0].set_xlabel(f'{numeric_cols[0]} (standardized)')
                    axes[1, 0].set_ylabel(f'{numeric_cols[1]} (standardized)')
                    axes[1, 0].set_title('DBSCAN Clustering')
                    plt.colorbar(scatter_db, ax=axes[1, 0])
                
                # Cluster silhouette analysis
                from sklearn.metrics import silhouette_score
                if len(set(kmeans_labels)) > 1:
                    silhouette_avg = silhouette_score(scaled_data, kmeans_labels)
                    clustering_results['kmeans']['silhouette_score'] = float(silhouette_avg)
                    
                    axes[1, 1].text(0.1, 0.5, f'K-means Silhouette Score: {silhouette_avg:.3f}', 
                                    transform=axes[1, 1].transAxes, fontsize=12)
                    
                if 'dbscan' in clustering_results and len(set(dbscan_labels)) > 1:
                    # Only calculate if there are actual clusters (not just noise)
                    if clustering_results['dbscan']['n_clusters'] > 1:
                        silhouette_db = silhouette_score(scaled_data, dbscan_labels)
                        clustering_results['dbscan']['silhouette_score'] = float(silhouette_db)
                        axes[1, 1].text(0.1, 0.3, f'DBSCAN Silhouette Score: {silhouette_db:.3f}', 
                                        transform=axes[1, 1].transAxes, fontsize=12)
                
                axes[1, 1].set_title('Clustering Quality Metrics')
                axes[1, 1].axis('off')
                
                plt.tight_layout()
                
                # Save plot
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                buffer.seek(0)
                plot_base64 = base64.b64encode(buffer.getvalue()).decode()
                buffer.close()
                plt.close()
                
                clustering_results['clustering_plots'] = plot_base64
                results['clustering_analysis'] = clustering_results
        
        return results
    
    def _perform_simple_regression(self, X, y, predictor_name, target_name):
        """Helper method for simple linear regression"""
        X_reshaped = X.values.reshape(-1, 1)
        
        # Fit regression
        reg = LinearRegression()
        reg.fit(X_reshaped, y)
        
        # Predictions
        y_pred = reg.predict(X_reshaped)
        
        # Calculate metrics
        r2 = r2_score(y, y_pred)
        
        # Statistical tests using statsmodels for p-values
        X_sm = sm.add_constant(X)
        model = sm.OLS(y, X_sm).fit()
        
        return {
            'predictor': predictor_name,
            'target': target_name,
            'coefficient': float(reg.coef_[0]),
            'intercept': float(reg.intercept_),
            'r_squared': float(r2),
            'p_value': float(model.pvalues[1]),  # p-value for coefficient
            'confidence_interval': model.conf_int().iloc[1].tolist(),
            'significant': model.pvalues[1] < 0.05,
            'residuals': (y - y_pred).tolist()
        }
    
    def _perform_multiple_regression(self, X, y, predictor_names, target_name):
        """Helper method for multiple linear regression"""
        # Fit regression
        reg = LinearRegression()
        reg.fit(X, y)
        
        # Predictions
        y_pred = reg.predict(X)
        
        # Calculate metrics
        r2 = r2_score(y, y_pred)
        
        # Statistical tests using statsmodels
        X_sm = sm.add_constant(X)
        model = sm.OLS(y, X_sm).fit()
        
        return {
            'predictors': predictor_names,
            'target': target_name,
            'coefficients': dict(zip(predictor_names, reg.coef_)),
            'intercept': float(reg.intercept_),
            'r_squared': float(r2),
            'adjusted_r_squared': float(model.rsquared_adj),
            'p_values': dict(zip(predictor_names, model.pvalues[1:])),
            'confidence_intervals': {name: model.conf_int().iloc[i+1].tolist() 
                                   for i, name in enumerate(predictor_names)},
            'f_statistic': float(model.fvalue),
            'f_p_value': float(model.f_pvalue),
            'residuals': (y - y_pred).tolist()
        }
    
    def _interpret_chi_square(self, p_value):
        """Interpret chi-square test results"""
        if p_value < 0.001:
            return "Highly significant association (p < 0.001)"
        elif p_value < 0.01:
            return "Significant association (p < 0.01)"
        elif p_value < 0.05:
            return "Significant association (p < 0.05)"
        else:
            return "No significant association (p ≥ 0.05)"
    
    def _interpret_p_value(self, p_value):
        """Generic p-value interpretation"""
        if p_value < 0.001:
            return "Highly significant (p < 0.001)"
        elif p_value < 0.01:
            return "Significant (p < 0.01)"
        elif p_value < 0.05:
            return "Significant (p < 0.05)"
        else:
            return "Not significant (p ≥ 0.05)"
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
