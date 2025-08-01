#!/usr/bin/env python3
"""
AutoStatIQ Command Line Interface (CLI)
Professional statistical analysis from the command line

Author: Roman Chaudhary
Contact: chaudharyroman.com.np
Version: 1.0.0

Usage:
    autostatiq-cli.py analyze data.csv
    autostatiq-cli.py analyze data.xlsx --question "Perform ANOVA analysis"
    autostatiq-cli.py batch *.csv --output results/
    autostatiq-cli.py health
    autostatiq-cli.py info
"""

import argparse
import sys
import os
import json
import glob
from pathlib import Path
from typing import List, Optional
import requests
from datetime import datetime
import time

class AutoStatIQCLI:
    """Command line interface for AutoStatIQ API"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:5000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({'X-API-Key': api_key})
        
        self.session.headers.update({
            'User-Agent': 'AutoStatIQ-CLI/1.0.0'
        })
    
    def health_check(self) -> bool:
        """Check API health status"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=30)
            response.raise_for_status()
            health = response.json()
            
            print("🏥 AutoStatIQ API Health Check")
            print("=" * 40)
            print(f"Status: {'✅ ' + health['status'].upper() if health['status'] == 'healthy' else '❌ ' + health['status'].upper()}")
            print(f"Version: {health.get('version', 'Unknown')}")
            print(f"Timestamp: {health.get('timestamp', 'Unknown')}")
            print(f"OpenAI Configured: {'✅ Yes' if health.get('openai_configured') else '❌ No'}")
            
            if health.get('features'):
                print(f"Features: {', '.join(health['features'])}")
            
            if health.get('supported_formats'):
                print(f"Supported Formats: {', '.join(health['supported_formats'])}")
            
            if health.get('missing_modules'):
                print(f"⚠️ Missing Modules: {', '.join(health['missing_modules'])}")
            
            return health['status'] == 'healthy'
            
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    def get_api_info(self) -> bool:
        """Get comprehensive API information"""
        try:
            response = self.session.get(f"{self.base_url}/api/info", timeout=30)
            response.raise_for_status()
            info = response.json()
            
            print("📊 AutoStatIQ API Information")
            print("=" * 40)
            print(f"Name: {info.get('name', 'AutoStatIQ API')}")
            print(f"Version: {info.get('version', 'Unknown')}")
            print(f"Description: {info.get('description', 'Statistical Analysis Platform')}")
            print(f"Author: {info.get('author', 'Unknown')}")
            print(f"Contact: {info.get('contact', 'Unknown')}")
            print(f"Base URL: {info.get('base_url', self.base_url)}")
            print()
            
            if info.get('statistical_methods'):
                print(f"📈 Statistical Methods ({len(info['statistical_methods'])}):")
                for i, method in enumerate(info['statistical_methods'], 1):
                    print(f"  {i:2d}. {method}")
                print()
            
            if info.get('control_chart_types'):
                print(f"📊 Control Chart Types ({len(info['control_chart_types'])}):")
                for i, chart_type in enumerate(info['control_chart_types'], 1):
                    print(f"  {i:2d}. {chart_type}")
                print()
            
            if info.get('endpoints'):
                print("🔗 API Endpoints:")
                for name, endpoint in info['endpoints'].items():
                    print(f"  {name}: {endpoint}")
                print()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to get API info: {e}")
            return False
    
    def analyze_file(self, file_path: str, question: Optional[str] = None, 
                    output_dir: Optional[str] = None, save_plots: bool = False) -> bool:
        """Analyze a single file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return False
        
        print(f"🔍 Analyzing: {file_path.name}")
        if question:
            print(f"❓ Question: {question}")
        
        try:
            with open(file_path, 'rb') as file:
                files = {'file': (file_path.name, file)}
                data = {'question': question} if question else {}
                
                print("⏳ Uploading and analyzing...")
                response = self.session.post(
                    f"{self.base_url}/upload",
                    files=files,
                    data=data,
                    timeout=300
                )
                response.raise_for_status()
                results = response.json()
            
            if results.get('success'):
                print("✅ Analysis completed successfully!")
                
                # Print summary
                if results.get('results'):
                    print("\n📊 Analysis Summary:")
                    self._print_results_summary(results['results'])
                
                # Print interpretation
                if results.get('interpretation'):
                    print("\n🧠 AI Interpretation:")
                    print(self._wrap_text(results['interpretation']))
                
                # Print conclusion
                if results.get('conclusion'):
                    print("\n🎯 Conclusion:")
                    print(self._wrap_text(results['conclusion']))
                
                # Save outputs
                if output_dir:
                    output_path = Path(output_dir)
                    output_path.mkdir(parents=True, exist_ok=True)
                    
                    # Save JSON results
                    json_file = output_path / f"{file_path.stem}_results.json"
                    with open(json_file, 'w') as f:
                        json.dump(results, f, indent=2)
                    print(f"💾 Results saved to: {json_file}")
                    
                    # Download report
                    if results.get('report_filename'):
                        report_path = output_path / f"{file_path.stem}_report.docx"
                        if self._download_report(results['report_filename'], report_path):
                            print(f"📄 Report saved to: {report_path}")
                
                return True
            else:
                print(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return False
    
    def batch_analyze(self, file_patterns: List[str], questions: Optional[List[str]] = None,
                     output_dir: Optional[str] = None) -> bool:
        """Analyze multiple files in batch"""
        # Expand file patterns
        files = []
        for pattern in file_patterns:
            files.extend(glob.glob(pattern))
        
        if not files:
            print("❌ No files found matching the patterns")
            return False
        
        files = [Path(f) for f in files if Path(f).is_file()]
        
        print(f"📁 Found {len(files)} files to analyze")
        
        questions = questions or [None] * len(files)
        success_count = 0
        
        for i, (file_path, question) in enumerate(zip(files, questions), 1):
            print(f"\n[{i}/{len(files)}] Processing: {file_path.name}")
            
            if self.analyze_file(file_path, question, output_dir):
                success_count += 1
            
            # Add delay between requests
            if i < len(files):
                time.sleep(1)
        
        print(f"\n📊 Batch Analysis Complete: {success_count}/{len(files)} successful")
        return success_count > 0
    
    def _download_report(self, report_filename: str, save_path: Path) -> bool:
        """Download report file"""
        try:
            response = self.session.get(f"{self.base_url}/download/{report_filename}")
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            return True
        except Exception as e:
            print(f"⚠️ Failed to download report: {e}")
            return False
    
    def _print_results_summary(self, results: dict):
        """Print a summary of analysis results"""
        summary_items = []
        
        # Count different types of results
        if 'descriptive_stats' in results:
            summary_items.append("Descriptive Statistics")
        
        if 'correlation_matrix' in results:
            summary_items.append("Correlation Analysis")
        
        if 'linear_regression' in results:
            summary_items.append("Regression Analysis")
        
        if 't_test' in results:
            summary_items.append("T-Test")
        
        if 'anova' in results:
            summary_items.append("ANOVA")
        
        if 'control_charts' in results:
            chart_count = len(results['control_charts'])
            summary_items.append(f"Control Charts ({chart_count})")
        
        if 'advanced_statistics' in results:
            summary_items.append("Advanced Statistics")
        
        if 'plots' in results:
            plot_count = len(results['plots'])
            summary_items.append(f"Visualizations ({plot_count})")
        
        if summary_items:
            for item in summary_items:
                print(f"  ✓ {item}")
        else:
            print("  No specific analysis results found")
    
    def _wrap_text(self, text: str, width: int = 80) -> str:
        """Wrap text to specified width"""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return '\n'.join(lines)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="AutoStatIQ Command Line Interface - Statistical Analysis in Sec",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s health                           # Check API health
  %(prog)s info                             # Get API information
  %(prog)s analyze data.csv                 # Analyze single file
  %(prog)s analyze data.xlsx -q "ANOVA"     # Analyze with question
  %(prog)s analyze data.csv -o results/     # Save outputs to directory
  %(prog)s batch *.csv                      # Analyze all CSV files
  %(prog)s batch data1.csv data2.xlsx -o results/

Author: Roman Chaudhary (chaudharyroman.com.np)
        """
    )
    
    parser.add_argument('--url', default='http://127.0.0.1:5000',
                       help='AutoStatIQ API base URL (default: http://127.0.0.1:5000)')
    parser.add_argument('--api-key', help='API key for authentication')
    parser.add_argument('--timeout', type=int, default=300,
                       help='Request timeout in seconds (default: 300)')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Health command
    health_parser = subparsers.add_parser('health', help='Check API health status')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Get API information')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a single file')
    analyze_parser.add_argument('file', help='File to analyze')
    analyze_parser.add_argument('-q', '--question', help='Statistical question')
    analyze_parser.add_argument('-o', '--output', help='Output directory for results')
    analyze_parser.add_argument('--save-plots', action='store_true',
                               help='Save visualization plots')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Analyze multiple files')
    batch_parser.add_argument('files', nargs='+', help='Files or patterns to analyze')
    batch_parser.add_argument('-q', '--questions', nargs='*',
                             help='Questions for each file (same order)')
    batch_parser.add_argument('-o', '--output', help='Output directory for results')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Initialize CLI client
    cli = AutoStatIQCLI(args.url, args.api_key)
    
    print("🧮 AutoStatIQ CLI - Statistical Analysis in Sec")
    print("=" * 50)
    
    # Execute command
    if args.command == 'health':
        success = cli.health_check()
    elif args.command == 'info':
        success = cli.get_api_info()
    elif args.command == 'analyze':
        success = cli.analyze_file(args.file, args.question, args.output, args.save_plots)
    elif args.command == 'batch':
        success = cli.batch_analyze(args.files, args.questions, args.output)
    else:
        print(f"❌ Unknown command: {args.command}")
        return 1
    
    return 0 if success else 1


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️ Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        sys.exit(1)
