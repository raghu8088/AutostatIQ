"""
Statistical Process Control (SPC) and Control Chart Module for AutoStatIQ
Supports various control chart types for quality control analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import base64
import io
from typing import Dict, List, Tuple, Optional, Any
import warnings

# Set matplotlib to use Agg backend for server environments
plt.switch_backend('Agg')

class ControlChartAnalyzer:
    """
    Comprehensive Statistical Process Control (SPC) analyzer
    Supports multiple control chart types for quality control
    """
    
    def __init__(self):
        self.control_limits = {}
        self.chart_data = {}
        self.violations = {}
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def detect_control_chart_needs(self, data: pd.DataFrame, question: str = "") -> Dict[str, Any]:
        """
        Automatically detect which control charts are appropriate for the data
        """
        recommendations = {
            'recommended_charts': [],
            'data_type': None,
            'sample_structure': None,
            'reasoning': []
        }
        
        # Analyze question for SPC keywords
        spc_keywords = [
            'control chart', 'spc', 'process control', 'quality control',
            'x-bar', 'r chart', 'p chart', 'c chart', 'u chart', 'i-mr',
            'individual', 'moving range', 'defects', 'nonconforming',
            'process capability', 'control limits', 'out of control'
        ]
        
        question_lower = question.lower()
        has_spc_keywords = any(keyword in question_lower for keyword in spc_keywords)
        
        # Analyze data structure
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return recommendations
            
        # Check for time series or sequential data
        has_time_column = any(col.lower() in ['time', 'date', 'sequence', 'order', 'sample'] 
                             for col in data.columns)
        
        # Detect data patterns
        for col in numeric_cols:
            values = data[col].dropna()
            
            if len(values) < 5:
                continue
                
            # Check for binary/count data (0,1 or small integers)
            is_binary = set(values.unique()).issubset({0, 1})
            is_count = all(val >= 0 and val == int(val) for val in values if not pd.isna(val))
            
            # Check for grouped data (subgroups)
            potential_subgroups = self._detect_subgroups(data, col)
            
            if is_binary or (is_count and values.max() <= 1):
                # P-chart for proportion defective
                recommendations['recommended_charts'].append({
                    'type': 'p_chart',
                    'column': col,
                    'description': 'P-chart for proportion of nonconforming items'
                })
                recommendations['reasoning'].append(f"Binary/proportion data detected in {col}")
                
            elif is_count and values.max() <= 20 and values.min() >= 0:
                # C-chart or U-chart for defects
                recommendations['recommended_charts'].append({
                    'type': 'c_chart',
                    'column': col,
                    'description': 'C-chart for count of defects'
                })
                recommendations['reasoning'].append(f"Count/defect data detected in {col}")
                
            elif potential_subgroups and len(potential_subgroups) > 2:
                # X-bar and R charts for subgrouped data
                recommendations['recommended_charts'].extend([
                    {
                        'type': 'xbar_chart',
                        'column': col,
                        'subgroups': potential_subgroups,
                        'description': 'X̄-chart for subgroup means'
                    },
                    {
                        'type': 'r_chart',
                        'column': col,
                        'subgroups': potential_subgroups,
                        'description': 'R-chart for subgroup ranges'
                    }
                ])
                recommendations['reasoning'].append(f"Subgrouped data detected for {col}")
                
            else:
                # I-MR charts for individual measurements
                recommendations['recommended_charts'].extend([
                    {
                        'type': 'i_chart',
                        'column': col,
                        'description': 'I-chart for individual measurements'
                    },
                    {
                        'type': 'mr_chart',
                        'column': col,
                        'description': 'MR-chart for moving ranges'
                    }
                ])
                recommendations['reasoning'].append(f"Individual measurements detected for {col}")
        
        # If SPC keywords found, prioritize control charts
        if has_spc_keywords:
            recommendations['reasoning'].insert(0, "SPC/Control chart analysis requested")
            
        recommendations['data_type'] = 'time_series' if has_time_column else 'sequential'
        
        return recommendations
    
    def _detect_subgroups(self, data: pd.DataFrame, target_col: str) -> Optional[List]:
        """
        Detect potential subgroup structure in data
        """
        # Look for grouping columns
        potential_group_cols = []
        for col in data.columns:
            if col != target_col and data[col].dtype in ['object', 'category']:
                unique_vals = data[col].nunique()
                if 2 <= unique_vals <= len(data) // 3:  # Reasonable number of groups
                    potential_group_cols.append(col)
        
        if potential_group_cols:
            # Use the first suitable grouping column
            group_col = potential_group_cols[0]
            subgroups = []
            for group in data[group_col].unique():
                group_data = data[data[group_col] == group][target_col].dropna()
                if len(group_data) >= 2:  # Minimum subgroup size
                    subgroups.append(group_data.tolist())
            
            if len(subgroups) >= 5:  # Minimum number of subgroups
                return subgroups
                
        return None
    
    def create_i_chart(self, data: List[float], title: str = "Individual Chart") -> Tuple[plt.Figure, Dict]:
        """
        Create Individual (I) control chart
        """
        data = np.array(data)
        n = len(data)
        
        if n < 2:
            raise ValueError("Need at least 2 data points for I-chart")
        
        # Calculate center line and control limits
        x_bar = np.mean(data)
        
        # Calculate moving ranges
        moving_ranges = np.abs(np.diff(data))
        mr_bar = np.mean(moving_ranges)
        
        # Control limits for I-chart
        ucl = x_bar + 2.66 * mr_bar
        lcl = x_bar - 2.66 * mr_bar
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot data points
        ax.plot(range(1, n + 1), data, 'bo-', markersize=6, linewidth=1.5, label='Individual Values')
        
        # Plot control lines
        ax.axhline(y=x_bar, color='green', linestyle='-', linewidth=2, label=f'Center Line (X̄={x_bar:.3f})')
        ax.axhline(y=ucl, color='red', linestyle='--', linewidth=2, label=f'UCL={ucl:.3f}')
        ax.axhline(y=lcl, color='red', linestyle='--', linewidth=2, label=f'LCL={lcl:.3f}')
        
        # Identify out-of-control points
        ooc_points = []
        for i, value in enumerate(data):
            if value > ucl or value < lcl:
                ax.plot(i + 1, value, 'ro', markersize=10, markerfacecolor='red', markeredgecolor='darkred')
                ooc_points.append({'point': i + 1, 'value': value, 'type': 'Beyond control limits'})
        
        # Formatting
        ax.set_xlabel('Sample Number')
        ax.set_ylabel('Individual Value')
        ax.set_title(f'{title} - Individual Chart (I-Chart)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Statistics
        stats_dict = {
            'center_line': x_bar,
            'ucl': ucl,
            'lcl': lcl,
            'mr_bar': mr_bar,
            'out_of_control_points': ooc_points,
            'process_capability': self._calculate_process_capability(data, ucl, lcl)
        }
        
        plt.tight_layout()
        return fig, stats_dict
    
    def create_mr_chart(self, data: List[float], title: str = "Moving Range Chart") -> Tuple[plt.Figure, Dict]:
        """
        Create Moving Range (MR) control chart
        """
        data = np.array(data)
        
        if len(data) < 2:
            raise ValueError("Need at least 2 data points for MR-chart")
        
        # Calculate moving ranges
        moving_ranges = np.abs(np.diff(data))
        mr_bar = np.mean(moving_ranges)
        
        # Control limits for MR-chart
        ucl = 3.267 * mr_bar
        lcl = 0  # Lower control limit is 0 for range charts
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot moving ranges
        ax.plot(range(2, len(data) + 1), moving_ranges, 'bo-', markersize=6, linewidth=1.5, label='Moving Ranges')
        
        # Plot control lines
        ax.axhline(y=mr_bar, color='green', linestyle='-', linewidth=2, label=f'Center Line (MR̄={mr_bar:.3f})')
        ax.axhline(y=ucl, color='red', linestyle='--', linewidth=2, label=f'UCL={ucl:.3f}')
        ax.axhline(y=lcl, color='red', linestyle='--', linewidth=2, label=f'LCL={lcl:.3f}')
        
        # Identify out-of-control points
        ooc_points = []
        for i, value in enumerate(moving_ranges):
            if value > ucl:
                ax.plot(i + 2, value, 'ro', markersize=10, markerfacecolor='red', markeredgecolor='darkred')
                ooc_points.append({'point': i + 2, 'value': value, 'type': 'Beyond control limits'})
        
        # Formatting
        ax.set_xlabel('Sample Number')
        ax.set_ylabel('Moving Range')
        ax.set_title(f'{title} - Moving Range Chart (MR-Chart)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Statistics
        stats_dict = {
            'center_line': mr_bar,
            'ucl': ucl,
            'lcl': lcl,
            'out_of_control_points': ooc_points
        }
        
        plt.tight_layout()
        return fig, stats_dict
    
    def create_xbar_chart(self, subgroups: List[List[float]], title: str = "X-bar Chart") -> Tuple[plt.Figure, Dict]:
        """
        Create X-bar control chart for subgrouped data
        """
        if len(subgroups) < 2:
            raise ValueError("Need at least 2 subgroups for X-bar chart")
        
        # Calculate subgroup means and overall mean
        subgroup_means = [np.mean(subgroup) for subgroup in subgroups]
        x_double_bar = np.mean(subgroup_means)
        
        # Calculate subgroup ranges and average range
        subgroup_ranges = [np.max(subgroup) - np.min(subgroup) for subgroup in subgroups]
        r_bar = np.mean(subgroup_ranges)
        
        # Determine subgroup size (assume constant)
        n = len(subgroups[0])
        
        # A2 factors for different subgroup sizes
        A2_factors = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483, 7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308}
        A2 = A2_factors.get(n, 0.308)  # Default to n=10 if not found
        
        # Control limits
        ucl = x_double_bar + A2 * r_bar
        lcl = x_double_bar - A2 * r_bar
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot subgroup means
        ax.plot(range(1, len(subgroup_means) + 1), subgroup_means, 'bo-', markersize=6, linewidth=1.5, label='Subgroup Means')
        
        # Plot control lines
        ax.axhline(y=x_double_bar, color='green', linestyle='-', linewidth=2, label=f'Center Line (X̄̄={x_double_bar:.3f})')
        ax.axhline(y=ucl, color='red', linestyle='--', linewidth=2, label=f'UCL={ucl:.3f}')
        ax.axhline(y=lcl, color='red', linestyle='--', linewidth=2, label=f'LCL={lcl:.3f}')
        
        # Identify out-of-control points
        ooc_points = []
        for i, mean_val in enumerate(subgroup_means):
            if mean_val > ucl or mean_val < lcl:
                ax.plot(i + 1, mean_val, 'ro', markersize=10, markerfacecolor='red', markeredgecolor='darkred')
                ooc_points.append({'subgroup': i + 1, 'value': mean_val, 'type': 'Beyond control limits'})
        
        # Formatting
        ax.set_xlabel('Subgroup Number')
        ax.set_ylabel('Subgroup Mean')
        ax.set_title(f'{title} - X̄ Chart (n={n})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Statistics
        stats_dict = {
            'center_line': x_double_bar,
            'ucl': ucl,
            'lcl': lcl,
            'r_bar': r_bar,
            'subgroup_size': n,
            'out_of_control_points': ooc_points,
            'process_capability': self._calculate_process_capability(subgroup_means, ucl, lcl)
        }
        
        plt.tight_layout()
        return fig, stats_dict
    
    def create_r_chart(self, subgroups: List[List[float]], title: str = "Range Chart") -> Tuple[plt.Figure, Dict]:
        """
        Create Range (R) control chart for subgrouped data
        """
        if len(subgroups) < 2:
            raise ValueError("Need at least 2 subgroups for R chart")
        
        # Calculate subgroup ranges
        subgroup_ranges = [np.max(subgroup) - np.min(subgroup) for subgroup in subgroups]
        r_bar = np.mean(subgroup_ranges)
        
        # Determine subgroup size
        n = len(subgroups[0])
        
        # D3 and D4 factors for different subgroup sizes
        D3_factors = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223}
        D4_factors = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777}
        
        D3 = D3_factors.get(n, 0)
        D4 = D4_factors.get(n, 1.777)
        
        # Control limits
        ucl = D4 * r_bar
        lcl = D3 * r_bar
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot subgroup ranges
        ax.plot(range(1, len(subgroup_ranges) + 1), subgroup_ranges, 'bo-', markersize=6, linewidth=1.5, label='Subgroup Ranges')
        
        # Plot control lines
        ax.axhline(y=r_bar, color='green', linestyle='-', linewidth=2, label=f'Center Line (R̄={r_bar:.3f})')
        ax.axhline(y=ucl, color='red', linestyle='--', linewidth=2, label=f'UCL={ucl:.3f}')
        ax.axhline(y=lcl, color='red', linestyle='--', linewidth=2, label=f'LCL={lcl:.3f}')
        
        # Identify out-of-control points
        ooc_points = []
        for i, range_val in enumerate(subgroup_ranges):
            if range_val > ucl or range_val < lcl:
                ax.plot(i + 1, range_val, 'ro', markersize=10, markerfacecolor='red', markeredgecolor='darkred')
                ooc_points.append({'subgroup': i + 1, 'value': range_val, 'type': 'Beyond control limits'})
        
        # Formatting
        ax.set_xlabel('Subgroup Number')
        ax.set_ylabel('Subgroup Range')
        ax.set_title(f'{title} - R Chart (n={n})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Statistics
        stats_dict = {
            'center_line': r_bar,
            'ucl': ucl,
            'lcl': lcl,
            'subgroup_size': n,
            'out_of_control_points': ooc_points
        }
        
        plt.tight_layout()
        return fig, stats_dict
    
    def create_p_chart(self, defective_counts: List[int], sample_sizes: List[int], title: str = "P Chart") -> Tuple[plt.Figure, Dict]:
        """
        Create P control chart for proportion defective
        """
        if len(defective_counts) != len(sample_sizes):
            raise ValueError("Defective counts and sample sizes must have same length")
        
        # Calculate proportions
        proportions = [d/n if n > 0 else 0 for d, n in zip(defective_counts, sample_sizes)]
        
        # Calculate center line (average proportion)
        total_defective = sum(defective_counts)
        total_samples = sum(sample_sizes)
        p_bar = total_defective / total_samples if total_samples > 0 else 0
        
        # Calculate control limits (variable limits for different sample sizes)
        ucl_values = []
        lcl_values = []
        
        for n in sample_sizes:
            if n > 0:
                ucl = p_bar + 3 * np.sqrt(p_bar * (1 - p_bar) / n)
                lcl = p_bar - 3 * np.sqrt(p_bar * (1 - p_bar) / n)
                lcl = max(0, lcl)  # LCL cannot be negative
            else:
                ucl = lcl = p_bar
            
            ucl_values.append(ucl)
            lcl_values.append(lcl)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot proportions
        ax.plot(range(1, len(proportions) + 1), proportions, 'bo-', markersize=6, linewidth=1.5, label='Proportion Defective')
        
        # Plot control lines
        ax.axhline(y=p_bar, color='green', linestyle='-', linewidth=2, label=f'Center Line (p̄={p_bar:.3f})')
        ax.plot(range(1, len(ucl_values) + 1), ucl_values, 'r--', linewidth=2, label='UCL')
        ax.plot(range(1, len(lcl_values) + 1), lcl_values, 'r--', linewidth=2, label='LCL')
        
        # Identify out-of-control points
        ooc_points = []
        for i, (prop, ucl, lcl) in enumerate(zip(proportions, ucl_values, lcl_values)):
            if prop > ucl or prop < lcl:
                ax.plot(i + 1, prop, 'ro', markersize=10, markerfacecolor='red', markeredgecolor='darkred')
                ooc_points.append({'sample': i + 1, 'value': prop, 'type': 'Beyond control limits'})
        
        # Formatting
        ax.set_xlabel('Sample Number')
        ax.set_ylabel('Proportion Defective')
        ax.set_title(f'{title} - P Chart')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Statistics
        stats_dict = {
            'center_line': p_bar,
            'average_ucl': np.mean(ucl_values),
            'average_lcl': np.mean(lcl_values),
            'total_defective': total_defective,
            'total_samples': total_samples,
            'out_of_control_points': ooc_points
        }
        
        plt.tight_layout()
        return fig, stats_dict
    
    def create_c_chart(self, defect_counts: List[int], title: str = "C Chart") -> Tuple[plt.Figure, Dict]:
        """
        Create C control chart for count of defects
        """
        defects = np.array(defect_counts)
        
        # Calculate center line and control limits
        c_bar = np.mean(defects)
        ucl = c_bar + 3 * np.sqrt(c_bar)
        lcl = max(0, c_bar - 3 * np.sqrt(c_bar))  # LCL cannot be negative
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot defect counts
        ax.plot(range(1, len(defects) + 1), defects, 'bo-', markersize=6, linewidth=1.5, label='Defect Count')
        
        # Plot control lines
        ax.axhline(y=c_bar, color='green', linestyle='-', linewidth=2, label=f'Center Line (c̄={c_bar:.3f})')
        ax.axhline(y=ucl, color='red', linestyle='--', linewidth=2, label=f'UCL={ucl:.3f}')
        ax.axhline(y=lcl, color='red', linestyle='--', linewidth=2, label=f'LCL={lcl:.3f}')
        
        # Identify out-of-control points
        ooc_points = []
        for i, count in enumerate(defects):
            if count > ucl or count < lcl:
                ax.plot(i + 1, count, 'ro', markersize=10, markerfacecolor='red', markeredgecolor='darkred')
                ooc_points.append({'sample': i + 1, 'value': count, 'type': 'Beyond control limits'})
        
        # Formatting
        ax.set_xlabel('Sample Number')
        ax.set_ylabel('Number of Defects')
        ax.set_title(f'{title} - C Chart')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Statistics
        stats_dict = {
            'center_line': c_bar,
            'ucl': ucl,
            'lcl': lcl,
            'total_defects': np.sum(defects),
            'out_of_control_points': ooc_points
        }
        
        plt.tight_layout()
        return fig, stats_dict
    
    def _calculate_process_capability(self, data: List[float], ucl: float, lcl: float) -> Dict[str, float]:
        """
        Calculate basic process capability indices
        """
        data = np.array(data)
        sigma = np.std(data, ddof=1)
        
        # Cp (potential capability)
        cp = (ucl - lcl) / (6 * sigma) if sigma > 0 else 0
        
        # Basic capability metrics
        return {
            'cp': cp,
            'sigma': sigma,
            'mean': np.mean(data)
        }
    
    def analyze_control_chart_violations(self, data: List[float], ucl: float, lcl: float, center_line: float) -> Dict[str, List]:
        """
        Analyze control chart for various types of violations (Western Electric Rules)
        """
        violations = {
            'rule1': [],  # Point beyond control limits
            'rule2': [],  # 9 points in a row on same side of center line
            'rule3': [],  # 6 points in a row steadily increasing or decreasing
            'rule4': []   # 14 points in a row alternating up and down
        }
        
        data = np.array(data)
        
        # Rule 1: Points beyond control limits
        for i, value in enumerate(data):
            if value > ucl or value < lcl:
                violations['rule1'].append({'point': i + 1, 'value': value})
        
        # Rule 2: 9 points in a row on same side of center line
        if len(data) >= 9:
            for i in range(len(data) - 8):
                segment = data[i:i+9]
                if all(x > center_line for x in segment) or all(x < center_line for x in segment):
                    violations['rule2'].append({'start_point': i + 1, 'end_point': i + 9})
        
        # Rule 3: 6 points in a row steadily increasing or decreasing
        if len(data) >= 6:
            for i in range(len(data) - 5):
                segment = data[i:i+6]
                if all(segment[j] < segment[j+1] for j in range(5)) or \
                   all(segment[j] > segment[j+1] for j in range(5)):
                    violations['rule3'].append({'start_point': i + 1, 'end_point': i + 6})
        
        return violations
    
    def generate_control_chart_interpretation(self, chart_type: str, stats: Dict, violations: Dict = None) -> str:
        """
        Generate interpretation text for control charts
        """
        interpretations = []
        
        # Basic chart information
        if chart_type == 'i_chart':
            interpretations.append("Individual (I) Chart Analysis:")
            interpretations.append(f"- Process center line: {stats['center_line']:.3f}")
            interpretations.append(f"- Control limits: {stats['lcl']:.3f} to {stats['ucl']:.3f}")
            
        elif chart_type == 'mr_chart':
            interpretations.append("Moving Range (MR) Chart Analysis:")
            interpretations.append(f"- Average moving range: {stats['center_line']:.3f}")
            interpretations.append(f"- Upper control limit: {stats['ucl']:.3f}")
            
        elif chart_type == 'xbar_chart':
            interpretations.append("X̄ Chart Analysis:")
            interpretations.append(f"- Process average: {stats['center_line']:.3f}")
            interpretations.append(f"- Control limits: {stats['lcl']:.3f} to {stats['ucl']:.3f}")
            interpretations.append(f"- Subgroup size: {stats['subgroup_size']}")
            
        elif chart_type == 'r_chart':
            interpretations.append("Range (R) Chart Analysis:")
            interpretations.append(f"- Average range: {stats['center_line']:.3f}")
            interpretations.append(f"- Upper control limit: {stats['ucl']:.3f}")
            
        elif chart_type == 'p_chart':
            interpretations.append("P Chart Analysis:")
            interpretations.append(f"- Average proportion defective: {stats['center_line']:.3f}")
            interpretations.append(f"- Total defective items: {stats['total_defective']}")
            interpretations.append(f"- Total samples inspected: {stats['total_samples']}")
            
        elif chart_type == 'c_chart':
            interpretations.append("C Chart Analysis:")
            interpretations.append(f"- Average defects per unit: {stats['center_line']:.3f}")
            interpretations.append(f"- Total defects observed: {stats['total_defects']}")
        
        # Out-of-control analysis
        ooc_points = stats.get('out_of_control_points', [])
        if ooc_points:
            interpretations.append(f"\n⚠️  OUT-OF-CONTROL SIGNALS DETECTED:")
            interpretations.append(f"- {len(ooc_points)} points exceed control limits")
            interpretations.append("- Investigation required for special causes")
            interpretations.append("- Process is NOT in statistical control")
        else:
            interpretations.append(f"\n✅ PROCESS IN CONTROL:")
            interpretations.append("- All points within control limits")
            interpretations.append("- Process appears stable")
            interpretations.append("- Only common cause variation present")
        
        # Process capability
        if 'process_capability' in stats:
            cp = stats['process_capability'].get('cp', 0)
            if cp > 0:
                interpretations.append(f"\n📊 PROCESS CAPABILITY:")
                interpretations.append(f"- Cp index: {cp:.3f}")
                if cp >= 1.33:
                    interpretations.append("- Excellent capability (Cp ≥ 1.33)")
                elif cp >= 1.0:
                    interpretations.append("- Adequate capability (1.0 ≤ Cp < 1.33)")
                else:
                    interpretations.append("- Poor capability (Cp < 1.0) - Process improvement needed")
        
        return "\n".join(interpretations)
    
    def save_chart_as_base64(self, fig: plt.Figure) -> str:
        """
        Convert matplotlib figure to base64 string for embedding in reports
        """
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        plt.close(fig)
        return image_base64
