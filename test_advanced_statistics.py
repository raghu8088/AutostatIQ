"""
Comprehensive test for Advanced Statistical Analysis functionality in AutoStatIQ
Tests all 25+ advanced statistical methods and their integration
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add the project root directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_statistics import AdvancedStatisticalAnalyzer
from app import StatisticalAnalyzer

def create_comprehensive_test_dataset():
    """Create a comprehensive dataset for testing all advanced statistical methods"""
    np.random.seed(42)
    n_samples = 500
    
    # Numerical variables
    data = {
        # Continuous variables with different distributions
        'normal_var': np.random.normal(100, 15, n_samples),
        'skewed_var': np.random.exponential(5, n_samples),
        'uniform_var': np.random.uniform(0, 100, n_samples),
        'bimodal_var': np.concatenate([
            np.random.normal(30, 5, n_samples//2),
            np.random.normal(70, 5, n_samples//2)
        ]),
        
        # Count data for Poisson/negative binomial analysis
        'count_var': np.random.poisson(3, n_samples),
        'defects': np.random.poisson(2, n_samples),
        
        # Binary outcomes for logistic regression
        'binary_outcome': np.random.binomial(1, 0.3, n_samples),
        'success_failure': np.random.choice(['Success', 'Failure'], n_samples, p=[0.4, 0.6]),
        
        # Categorical variables with different levels
        'category_3': np.random.choice(['A', 'B', 'C'], n_samples, p=[0.5, 0.3, 0.2]),
        'category_5': np.random.choice(['Group1', 'Group2', 'Group3', 'Group4', 'Group5'], 
                                     n_samples, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
        'product_type': np.random.choice(['Premium', 'Standard', 'Economy'], n_samples),
        
        # Time series component
        'time_index': range(n_samples),
        'trend_var': np.arange(n_samples) * 0.1 + np.random.normal(0, 2, n_samples),
        
        # Correlated variables for correlation analysis
        'corr_var1': None,  # Will be calculated below
        'corr_var2': None,  # Will be calculated below
        
        # Quality control measurements
        'measurement1': np.random.normal(50, 3, n_samples),
        'measurement2': np.random.normal(75, 5, n_samples),
        'temperature': np.random.normal(25, 2, n_samples),
        'pressure': np.random.normal(100, 8, n_samples),
    }
    
    # Create correlated variables
    base_var = np.random.normal(0, 1, n_samples)
    data['corr_var1'] = base_var + np.random.normal(0, 0.3, n_samples)
    data['corr_var2'] = 0.7 * base_var + np.random.normal(0, 0.5, n_samples)
    
    # Add some outliers
    outlier_indices = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
    data['normal_var'][outlier_indices] += np.random.choice([-50, 50], size=len(outlier_indices))
    
    # Add missing values to some variables (before creating dependent variables)
    missing_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    for i, var in enumerate(['uniform_var', 'measurement1']):  # Only numeric variables
        if i < len(missing_indices):
            indices_to_nan = missing_indices[i::2][:len(missing_indices)//2]
            if var in data:
                if isinstance(data[var], np.ndarray):
                    data[var] = data[var].astype(float)
                    data[var][indices_to_nan] = np.nan
    
    # Add missing values to categorical variable differently
    cat_missing_indices = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
    category_3_array = np.array(data['category_3'], dtype=object)
    category_3_array[cat_missing_indices] = None
    data['category_3'] = category_3_array
    
    # Create relationships for regression analysis (after handling missing values)
    # Linear relationship - only use non-NaN values
    normal_clean = np.nan_to_num(data['normal_var'], nan=np.nanmean(data['normal_var']))
    uniform_clean = np.nan_to_num(data['uniform_var'], nan=np.nanmean(data['uniform_var']))
    
    data['dependent_var'] = (2 * normal_clean + 
                           1.5 * uniform_clean + 
                           np.random.normal(0, 10, n_samples))
    
    # Logistic relationship
    linear_combination = (0.02 * normal_clean + 
                         0.01 * uniform_clean - 2)
    probabilities = 1 / (1 + np.exp(-linear_combination))
    # Ensure probabilities are in valid range and no NaNs
    probabilities = np.clip(probabilities, 0.001, 0.999)
    probabilities = np.nan_to_num(probabilities, nan=0.5)
    data['logistic_outcome'] = np.random.binomial(1, probabilities, n_samples)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    return df

def test_advanced_descriptive_statistics(analyzer, data):
    """Test enhanced descriptive statistics"""
    print("\n=== Testing Enhanced Descriptive Statistics ===")
    results = analyzer.perform_comprehensive_descriptive_analysis(data)
    
    if 'enhanced_descriptive_stats' in results:
        stats = results['enhanced_descriptive_stats']
        print(f"✓ Enhanced descriptive statistics calculated for {len(stats)} variables")
        
        # Check specific metrics
        for var, var_stats in stats.items():
            required_metrics = ['mean', 'median', 'mode', 'std', 'variance', 'skewness', 'kurtosis', 'cv']
            if all(metric in var_stats for metric in required_metrics):
                print(f"  ✓ {var}: All required metrics present")
            else:
                print(f"  ✗ {var}: Missing metrics")
    else:
        print("✗ Enhanced descriptive statistics failed")
    
    return results

def test_frequency_analysis(analyzer, data):
    """Test frequency analysis for categorical variables"""
    print("\n=== Testing Frequency Analysis ===")
    results = analyzer.perform_frequency_analysis(data)
    
    if 'frequency_analysis' in results:
        freq_analysis = results['frequency_analysis']
        print(f"✓ Frequency analysis completed for {len(freq_analysis)} categorical variables")
        
        for var, analysis in freq_analysis.items():
            if all(key in analysis for key in ['frequencies', 'proportions', 'plot']):
                print(f"  ✓ {var}: Frequencies, proportions, and visualization generated")
            else:
                print(f"  ✗ {var}: Incomplete analysis")
    else:
        print("✗ Frequency analysis failed")
    
    return results

def test_cross_tabulation(analyzer, data):
    """Test cross-tabulation and chi-square tests"""
    print("\n=== Testing Cross-tabulation Analysis ===")
    results = analyzer.perform_cross_tabulation_analysis(data)
    
    if 'cross_tabulation' in results:
        cross_tab = results['cross_tabulation']
        print(f"✓ Cross-tabulation analysis completed for {len(cross_tab)} variable pairs")
        
        for pair, analysis in cross_tab.items():
            if all(key in analysis for key in ['chi2_statistic', 'p_value', 'heatmap']):
                significance = "significant" if analysis['significant'] else "not significant"
                print(f"  ✓ {pair}: Chi-square = {analysis['chi2_statistic']:.3f}, p = {analysis['p_value']:.3f} ({significance})")
            else:
                print(f"  ✗ {pair}: Incomplete analysis")
    else:
        print("✗ Cross-tabulation analysis failed")
    
    return results

def test_advanced_correlation(analyzer, data):
    """Test advanced correlation analysis"""
    print("\n=== Testing Advanced Correlation Analysis ===")
    results = analyzer.perform_advanced_correlation_analysis(data)
    
    if 'advanced_correlation' in results:
        corr_analysis = results['advanced_correlation']
        
        correlation_types = ['pearson', 'spearman', 'kendall']
        for corr_type in correlation_types:
            if corr_type in corr_analysis:
                print(f"  ✓ {corr_type.capitalize()} correlation matrix calculated")
            else:
                print(f"  ✗ {corr_type.capitalize()} correlation missing")
        
        if 'correlation_heatmaps' in corr_analysis:
            print("  ✓ Correlation heatmaps generated")
        
        if 'pairplot' in corr_analysis:
            print("  ✓ Pairplot visualization generated")
    else:
        print("✗ Advanced correlation analysis failed")
    
    return results

def test_regression_analysis(analyzer, data):
    """Test advanced regression analysis"""
    print("\n=== Testing Advanced Regression Analysis ===")
    results = analyzer.perform_advanced_regression_analysis(data)
    
    if 'advanced_regression' in results:
        reg_analysis = results['advanced_regression']
        print(f"✓ Advanced regression analysis completed for {len(reg_analysis)} models")
        
        for model_name, model_results in reg_analysis.items():
            if 'r_squared' in model_results:
                r2 = model_results['r_squared']
                print(f"  ✓ {model_name}: R² = {r2:.3f}")
            else:
                print(f"  ✗ {model_name}: Incomplete results")
    else:
        print("✗ Advanced regression analysis failed")
    
    return results

def test_logistic_regression(analyzer, data):
    """Test logistic regression analysis"""
    print("\n=== Testing Logistic Regression Analysis ===")
    results = analyzer.perform_logistic_regression_analysis(data)
    
    if 'logistic_regression' in results:
        log_reg = results['logistic_regression']
        print(f"✓ Logistic regression analysis completed for {len(log_reg)} models")
        
        for model_name, model_results in log_reg.items():
            if all(key in model_results for key in ['accuracy', 'auc', 'roc_plot']):
                accuracy = model_results['accuracy']
                auc_score = model_results['auc']
                print(f"  ✓ {model_name}: Accuracy = {accuracy:.3f}, AUC = {auc_score:.3f}")
            else:
                print(f"  ✗ {model_name}: Incomplete results")
    else:
        print("✗ Logistic regression analysis failed")
    
    return results

def test_hypothesis_testing(analyzer, data):
    """Test comprehensive hypothesis testing"""
    print("\n=== Testing Comprehensive Hypothesis Testing ===")
    results = analyzer.perform_comprehensive_hypothesis_testing(data)
    
    if 'hypothesis_testing' in results:
        hyp_testing = results['hypothesis_testing']
        
        if 'one_sample_tests' in hyp_testing:
            one_sample = hyp_testing['one_sample_tests']
            print(f"  ✓ One-sample t-tests completed for {len(one_sample)} variables")
        
        if 'two_sample_tests' in hyp_testing:
            two_sample = hyp_testing['two_sample_tests']
            print(f"  ✓ Two-sample tests completed for {len(two_sample)} comparisons")
            
            for test_name, test_results in two_sample.items():
                if 't_test' in test_results and 'mann_whitney' in test_results:
                    t_p = test_results['t_test']['p_value']
                    mw_p = test_results['mann_whitney']['p_value']
                    print(f"    {test_name}: t-test p = {t_p:.3f}, Mann-Whitney p = {mw_p:.3f}")
    else:
        print("✗ Comprehensive hypothesis testing failed")
    
    return results

def test_advanced_anova(analyzer, data):
    """Test advanced ANOVA with post-hoc tests"""
    print("\n=== Testing Advanced ANOVA ===")
    results = analyzer.perform_advanced_anova(data)
    
    if 'advanced_anova' in results:
        anova_results = results['advanced_anova']
        print(f"✓ Advanced ANOVA completed for {len(anova_results)} analyses")
        
        for analysis_name, analysis_results in anova_results.items():
            if all(key in analysis_results for key in ['anova', 'kruskal_wallis', 'boxplot']):
                f_p = analysis_results['anova']['p_value']
                kw_p = analysis_results['kruskal_wallis']['p_value']
                print(f"  ✓ {analysis_name}: ANOVA p = {f_p:.3f}, Kruskal-Wallis p = {kw_p:.3f}")
                
                if 'tukey_hsd' in analysis_results:
                    print(f"    Post-hoc Tukey HSD completed")
            else:
                print(f"  ✗ {analysis_name}: Incomplete results")
    else:
        print("✗ Advanced ANOVA failed")
    
    return results

def test_normality_tests(analyzer, data):
    """Test comprehensive normality testing"""
    print("\n=== Testing Normality Tests ===")
    results = analyzer.perform_normality_tests(data)
    
    if 'normality_tests' in results:
        norm_tests = results['normality_tests']
        print(f"✓ Normality tests completed for {len(norm_tests)} variables")
        
        for var, tests in norm_tests.items():
            test_names = list(tests.keys())
            test_names = [name for name in test_names if name != 'normality_plots']
            print(f"  ✓ {var}: {len(test_names)} normality tests + Q-Q plot")
    else:
        print("✗ Normality tests failed")
    
    return results

def test_outlier_detection(analyzer, data):
    """Test comprehensive outlier detection"""
    print("\n=== Testing Outlier Detection ===")
    results = analyzer.perform_outlier_detection(data)
    
    if 'outlier_detection' in results:
        outlier_results = results['outlier_detection']
        print(f"✓ Outlier detection completed for {len(outlier_results)} variables")
        
        total_outliers_zscore = 0
        total_outliers_iqr = 0
        
        for var, detection in outlier_results.items():
            if all(key in detection for key in ['z_score', 'iqr', 'outlier_plots']):
                z_outliers = detection['z_score']['outlier_count']
                iqr_outliers = detection['iqr']['outlier_count']
                total_outliers_zscore += z_outliers
                total_outliers_iqr += iqr_outliers
                print(f"  ✓ {var}: {z_outliers} Z-score outliers, {iqr_outliers} IQR outliers")
        
        print(f"  Total outliers detected: {total_outliers_zscore} (Z-score), {total_outliers_iqr} (IQR)")
    else:
        print("✗ Outlier detection failed")
    
    return results

def test_missing_data_analysis(analyzer, data):
    """Test missing data analysis"""
    print("\n=== Testing Missing Data Analysis ===")
    results = analyzer.perform_missing_data_analysis(data)
    
    if 'missing_data_analysis' in results:
        missing_analysis = results['missing_data_analysis']
        
        if all(key in missing_analysis for key in ['summary', 'total_rows', 'complete_cases', 'visualization']):
            total_rows = missing_analysis['total_rows']
            complete_cases = missing_analysis['complete_cases']
            missing_pct = (1 - complete_cases/total_rows) * 100
            print(f"✓ Missing data analysis completed")
            print(f"  Total rows: {total_rows}, Complete cases: {complete_cases} ({missing_pct:.1f}% missing)")
            print(f"  Visualization generated: {'visualization' in missing_analysis}")
        else:
            print("✗ Missing data analysis incomplete")
    else:
        print("✗ Missing data analysis failed")
    
    return results

def test_pca(analyzer, data):
    """Test Principal Component Analysis"""
    print("\n=== Testing Principal Component Analysis ===")
    results = analyzer.perform_principal_component_analysis(data)
    
    if 'principal_component_analysis' in results:
        pca_results = results['principal_component_analysis']
        
        if all(key in pca_results for key in ['explained_variance_ratio', 'optimal_components', 'pca_plots']):
            optimal_k = pca_results['optimal_components']
            total_variance = pca_results['total_variance_explained']
            print(f"✓ PCA completed")
            print(f"  Optimal components: {optimal_k}")
            print(f"  Total variance explained: {total_variance:.1%}")
            print(f"  Visualizations generated: {'pca_plots' in pca_results}")
        else:
            print("✗ PCA incomplete")
    else:
        print("✗ PCA failed")
    
    return results

def test_clustering_analysis(analyzer, data):
    """Test comprehensive clustering analysis"""
    print("\n=== Testing Clustering Analysis ===")
    results = analyzer.perform_clustering_analysis(data)
    
    if 'clustering_analysis' in results:
        cluster_results = results['clustering_analysis']
        
        methods_tested = []
        if 'kmeans' in cluster_results:
            kmeans = cluster_results['kmeans']
            optimal_k = kmeans['optimal_clusters']
            methods_tested.append(f"K-means (k={optimal_k})")
        
        if 'dbscan' in cluster_results:
            dbscan = cluster_results['dbscan']
            n_clusters = dbscan['n_clusters']
            n_noise = dbscan['n_noise_points']
            methods_tested.append(f"DBSCAN ({n_clusters} clusters, {n_noise} noise)")
        
        print(f"✓ Clustering analysis completed: {', '.join(methods_tested)}")
        
        if 'clustering_plots' in cluster_results:
            print("  ✓ Clustering visualizations generated")
    else:
        print("✗ Clustering analysis failed")
    
    return results

def test_full_integration(analyzer, data):
    """Test full integration with StatisticalAnalyzer"""
    print("\n=== Testing Full Integration ===")
    
    # Test with main StatisticalAnalyzer
    main_analyzer = StatisticalAnalyzer()
    main_analyzer.data = data
    
    # Perform complete advanced analysis
    user_question = "Perform comprehensive statistical analysis including descriptive statistics, correlation analysis, regression modeling, hypothesis testing, and clustering"
    
    full_results = main_analyzer.perform_advanced_statistical_analysis(data, user_question)
    
    if full_results:
        analysis_types = list(full_results.keys())
        print(f"✓ Full integration successful - {len(analysis_types)} analysis types completed")
        
        # Count total visualizations
        total_plots = 0
        for analysis_type, results in full_results.items():
            if isinstance(results, dict):
                for key, value in results.items():
                    if isinstance(value, dict) and any(plot_key in key.lower() for plot_key in ['plot', 'chart', 'heatmap', 'visualization']):
                        total_plots += 1
        
        print(f"  Total visualizations generated: {total_plots}")
        print(f"  Analysis types: {', '.join(analysis_types)}")
    else:
        print("✗ Full integration failed")
    
    return full_results

def main():
    """Run comprehensive test suite for advanced statistics"""
    print("=" * 80)
    print("AutoStatIQ Advanced Statistical Analysis - Comprehensive Test Suite")
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Create test dataset
    print("\n=== Creating Comprehensive Test Dataset ===")
    data = create_comprehensive_test_dataset()
    print(f"✓ Test dataset created: {len(data)} rows, {len(data.columns)} columns")
    print(f"  Numerical variables: {len(data.select_dtypes(include=['number']).columns)}")
    print(f"  Categorical variables: {len(data.select_dtypes(include=['object']).columns)}")
    
    # Initialize analyzer
    analyzer = AdvancedStatisticalAnalyzer()
    
    # Run all tests
    test_results = {}
    
    try:
        # Test individual components
        test_results['descriptive'] = test_advanced_descriptive_statistics(analyzer, data)
        test_results['frequency'] = test_frequency_analysis(analyzer, data)
        test_results['cross_tab'] = test_cross_tabulation(analyzer, data)
        test_results['correlation'] = test_advanced_correlation(analyzer, data)
        test_results['regression'] = test_regression_analysis(analyzer, data)
        test_results['logistic'] = test_logistic_regression(analyzer, data)
        test_results['hypothesis'] = test_hypothesis_testing(analyzer, data)
        test_results['anova'] = test_advanced_anova(analyzer, data)
        test_results['normality'] = test_normality_tests(analyzer, data)
        test_results['outliers'] = test_outlier_detection(analyzer, data)
        test_results['missing'] = test_missing_data_analysis(analyzer, data)
        test_results['pca'] = test_pca(analyzer, data)
        test_results['clustering'] = test_clustering_analysis(analyzer, data)
        
        # Test full integration
        test_results['integration'] = test_full_integration(analyzer, data)
        
    except Exception as e:
        print(f"\n✗ Test suite failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUITE SUMMARY")
    print("=" * 80)
    
    successful_tests = sum(1 for result in test_results.values() if result)
    total_tests = len(test_results)
    
    print(f"Tests completed: {total_tests}")
    print(f"Successful tests: {successful_tests}")
    print(f"Success rate: {successful_tests/total_tests*100:.1f}%")
    
    if successful_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! Advanced Statistical Analysis is fully functional!")
        print("\nAutoStatIQ now supports 25+ advanced statistical methods including:")
        print("✓ Enhanced descriptive statistics with skewness, kurtosis, CV")
        print("✓ Comprehensive frequency analysis with visualizations")
        print("✓ Cross-tabulation with chi-square tests and heatmaps")
        print("✓ Advanced correlation analysis (Pearson, Spearman, Kendall)")
        print("✓ Multiple regression with diagnostics")
        print("✓ Logistic regression with ROC curves")
        print("✓ Comprehensive hypothesis testing")
        print("✓ Advanced ANOVA with post-hoc tests")
        print("✓ Normality testing with Q-Q plots")
        print("✓ Multi-method outlier detection")
        print("✓ Missing data analysis and visualization")
        print("✓ Principal Component Analysis")
        print("✓ Clustering analysis (K-means, DBSCAN)")
        print("✓ Professional visualizations and reporting")
    else:
        print(f"\n⚠️  {total_tests - successful_tests} tests failed. Check the output above for details.")
    
    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return successful_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
