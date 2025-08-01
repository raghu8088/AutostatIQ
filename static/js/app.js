// AutoStatIQ JavaScript Application

class AutoStatIQ {
    constructor() {
        this.dropzone = null;
        this.currentReportFilename = null;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupDropzone();
        this.setupFormValidation();
    }

    setupEventListeners() {
        // Form submission
        document.getElementById('uploadForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleFormSubmission();
        });

        // Download button
        document.getElementById('downloadBtn').addEventListener('click', () => {
            this.downloadReport();
        });

        // File input change
        document.getElementById('fileInput').addEventListener('change', (e) => {
            this.handleFileSelection(e);
        });

        // Question-only analysis
        document.getElementById('question').addEventListener('input', (e) => {
            this.toggleAnalysisMode(e.target.value.trim());
        });
    }

    setupDropzone() {
        // Initialize Dropzone
        Dropzone.autoDiscover = false;
        
        const dropzoneElement = document.getElementById('dropzone');
        
        this.dropzone = new Dropzone(dropzoneElement, {
            url: '/upload', // This won't be used as we handle submission manually
            autoProcessQueue: false,
            maxFiles: 1,
            maxFilesize: 16, // 16MB
            acceptedFiles: '.csv,.xlsx,.xls,.docx,.pdf,.json,.txt',
            addRemoveLinks: true,
            dictDefaultMessage: `
                <i class="fas fa-cloud-upload-alt fa-3x text-muted mb-3"></i>
                <h5>Drop files here or click to upload</h5>
                <p class="text-muted">Supports CSV, XLSX, DOCX, PDF, JSON, TXT files (max 16MB)</p>
            `,
            previewTemplate: this.getDropzonePreviewTemplate()
        });

        // Dropzone event handlers
        this.dropzone.on('addedfile', (file) => {
            this.handleDropzoneFileAdded(file);
        });

        this.dropzone.on('removedfile', (file) => {
            this.handleDropzoneFileRemoved(file);
        });

        this.dropzone.on('error', (file, message) => {
            this.showError(`File upload error: ${message}`);
        });

        // Custom click handler
        dropzoneElement.addEventListener('click', (e) => {
            if (e.target.closest('.dz-remove')) return;
            document.getElementById('fileInput').click();
        });
    }

    getDropzonePreviewTemplate() {
        return `
            <div class="dz-preview dz-file-preview">
                <div class="dz-details">
                    <div class="dz-filename"><span data-dz-name></span></div>
                    <div class="dz-size" data-dz-size></div>
                </div>
                <div class="dz-progress"><span class="dz-upload" data-dz-uploadprogress></span></div>
                <div class="dz-success-mark"><span>✓</span></div>
                <div class="dz-error-mark"><span>✗</span></div>
                <div class="dz-error-message"><span data-dz-errormessage></span></div>
                <a class="dz-remove" href="javascript:undefined;" data-dz-remove>Remove file</a>
            </div>
        `;
    }

    setupFormValidation() {
        const form = document.getElementById('uploadForm');
        const questionInput = document.getElementById('question');
        const analyzeBtn = document.getElementById('analyzeBtn');

        // Real-time validation
        form.addEventListener('input', () => {
            this.validateForm();
        });
    }

    validateForm() {
        const question = document.getElementById('question').value.trim();
        const hasFile = this.dropzone.files.length > 0;
        const analyzeBtn = document.getElementById('analyzeBtn');

        if (question || hasFile) {
            analyzeBtn.disabled = false;
            analyzeBtn.classList.remove('btn-secondary');
            analyzeBtn.classList.add('btn-primary');
        } else {
            analyzeBtn.disabled = true;
            analyzeBtn.classList.remove('btn-primary');
            analyzeBtn.classList.add('btn-secondary');
        }
    }

    toggleAnalysisMode(hasQuestion) {
        const dropzoneContainer = document.getElementById('dropzone').parentElement;
        
        if (hasQuestion) {
            dropzoneContainer.style.opacity = '0.7';
            document.querySelector('[for="question"]').innerHTML = `
                <i class="fas fa-question-circle me-1"></i>
                Statistical Question <span class="badge bg-success">Ready for Analysis</span>
            `;
        } else {
            dropzoneContainer.style.opacity = '1';
            document.querySelector('[for="question"]').innerHTML = `
                <i class="fas fa-question-circle me-1"></i>
                Statistical Question (Optional)
            `;
        }
    }

    handleFileSelection(event) {
        const file = event.target.files[0];
        if (file) {
            // Clear existing files and add new one
            this.dropzone.removeAllFiles();
            this.dropzone.addFile(file);
        }
    }

    handleDropzoneFileAdded(file) {
        this.validateForm();
        this.showSuccess(`File "${file.name}" ready for analysis`);
    }

    handleDropzoneFileRemoved(file) {
        this.validateForm();
        document.getElementById('fileInput').value = '';
    }

    async handleFormSubmission() {
        const question = document.getElementById('question').value.trim();
        const hasFile = this.dropzone.files.length > 0;

        // Validate input
        if (!question && !hasFile) {
            this.showError('Please provide either a question or upload a file for analysis');
            return;
        }

        // Show loading state
        this.showLoading();

        try {
            let response;

            if (hasFile) {
                // File upload analysis
                response = await this.uploadFileForAnalysis(question);
            } else {
                // Question-only analysis
                response = await this.analyzeQuestionOnly(question);
            }

            if (response.success) {
                this.displayResults(response);
                this.currentReportFilename = response.report_filename;
            } else {
                throw new Error(response.error || 'Analysis failed');
            }

        } catch (error) {
            this.showError(error.message);
        } finally {
            this.hideLoading();
        }
    }

    async uploadFileForAnalysis(question) {
        const formData = new FormData();
        
        if (this.dropzone.files.length > 0) {
            formData.append('file', this.dropzone.files[0]);
        }
        
        if (question) {
            formData.append('question', question);
        }

        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        return await response.json();
    }

    async analyzeQuestionOnly(question) {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question: question })
        });

        return await response.json();
    }

    displayResults(response) {
        const resultsCard = document.getElementById('resultsCard');
        
        // Show results card
        resultsCard.classList.remove('d-none');
        resultsCard.scrollIntoView({ behavior: 'smooth' });

        // Populate content
        this.populateSummary(response.results);
        this.populateInterpretation(response.interpretation);
        
        // Handle all types of visualizations
        const allVisualizations = { ...response.plots };
        
        // Add control charts to visualizations
        if (response.results.control_charts) {
            for (const [chartKey, chartData] of Object.entries(response.results.control_charts)) {
                allVisualizations[chartKey] = chartData;
            }
        }
        
        // Add advanced statistics visualizations
        this.extractAdvancedVisualizationsFromResults(response.results, allVisualizations);
        
        this.populateVisualizations(allVisualizations);
        this.populateConclusion(response.conclusion);

        // Show download button if report is available
        if (response.report_filename) {
            document.getElementById('downloadBtn').classList.remove('d-none');
        }

        // Add animations
        resultsCard.classList.add('fade-in-up');
    }

    extractAdvancedVisualizationsFromResults(results, visualizations) {
        // Extract visualizations from advanced statistics results
        const advancedTypes = [
            'frequency_analysis',
            'cross_tabulation', 
            'advanced_correlation',
            'advanced_regression',
            'logistic_regression',
            'advanced_anova',
            'normality_tests',
            'outlier_detection',
            'missing_data_analysis',
            'principal_component_analysis',
            'clustering_analysis'
        ];
        
        for (const type of advancedTypes) {
            if (results[type]) {
                this.extractPlotsFromAnalysisType(results[type], type, visualizations);
            }
        }
    }
    
    extractPlotsFromAnalysisType(analysisData, analysisType, visualizations) {
        // Recursively look for plot data in analysis results
        if (Array.isArray(analysisData)) {
            analysisData.forEach((item, index) => {
                this.extractPlotsFromAnalysisType(item, `${analysisType}_${index}`, visualizations);
            });
        } else if (typeof analysisData === 'object' && analysisData !== null) {
            for (const [key, value] of Object.entries(analysisData)) {
                if (key.includes('plot') || key.includes('heatmap') || key.includes('chart') || 
                    key === 'visualization' || key === 'pca_plots' || key === 'clustering_plots' ||
                    key === 'correlation_heatmaps' || key === 'pairplot' || key === 'roc_plot' ||
                    key === 'boxplot' || key === 'normality_plots' || key === 'outlier_plots') {
                    visualizations[`${analysisType}_${key}`] = {
                        chart_image: value,
                        type: `${analysisType} - ${key}`,
                        analysis_type: analysisType
                    };
                } else if (typeof value === 'object' && value !== null) {
                    this.extractPlotsFromAnalysisType(value, `${analysisType}_${key}`, visualizations);
                }
            }
        }
    }

    populateSummary(results) {
        const summaryContent = document.getElementById('summaryContent');
        let html = '';

        // Organize results by category
        const categories = {
            'Descriptive Statistics': ['descriptive_stats', 'enhanced_descriptive_stats'],
            'Frequency & Cross-tabulation': ['categorical_analysis', 'frequency_analysis', 'cross_tabulation'],
            'Correlation & Regression': ['correlation_matrix', 'advanced_correlation', 'advanced_regression', 'logistic_regression'],
            'Hypothesis Testing': ['t_test', 'correlation_test', 'anova', 'hypothesis_testing', 'advanced_anova'],
            'Data Quality': ['normality_tests', 'outlier_detection', 'missing_data_analysis'],
            'Advanced Analytics': ['principal_component_analysis', 'clustering_analysis'],
            'Process Control': ['control_charts', 'control_chart_recommendations'],
            'Other Analysis': []
        };

        // Add uncategorized results to "Other Analysis"
        const allCategorized = new Set();
        Object.values(categories).forEach(cat => cat.forEach(item => allCategorized.add(item)));
        
        for (const key of Object.keys(results)) {
            if (!allCategorized.has(key) && key !== 'plots') {
                categories['Other Analysis'].push(key);
            }
        }

        // Generate HTML for each category
        for (const [categoryName, categoryKeys] of Object.entries(categories)) {
            const categoryResults = categoryKeys.filter(key => results[key]);
            
            if (categoryResults.length > 0) {
                html += `
                    <div class="results-category mb-4">
                        <h4 class="category-title text-primary border-bottom pb-2">
                            <i class="fas fa-chart-line me-2"></i>
                            ${categoryName}
                        </h4>
                `;
                
                for (const key of categoryResults) {
                    html += `
                        <div class="result-item mb-3 p-3 bg-light rounded">
                            <h6 class="result-title text-secondary">
                                ${this.formatTitle(key)}
                            </h6>
                            <div class="results-content">
                                ${this.formatResultValue(results[key])}
                            </div>
                        </div>
                    `;
                }
                
                html += `</div>`;
            }
        }

        summaryContent.innerHTML = html || '<p class="text-muted">No summary data available.</p>';
    }

    populateInterpretation(interpretation) {
        const interpretationContent = document.getElementById('interpretationContent');
        interpretationContent.innerHTML = `
            <div class="interpretation-text">
                <div class="alert alert-info">
                    <i class="fas fa-lightbulb me-2"></i>
                    <strong>AI Interpretation:</strong>
                </div>
                <div class="formatted-text">
                    ${this.formatText(interpretation)}
                </div>
            </div>
        `;
    }

    populateVisualizations(plots) {
        const visualizationsContent = document.getElementById('visualizationsContent');
        
        if (!plots || Object.keys(plots).length === 0) {
            visualizationsContent.innerHTML = `
                <div class="text-center text-muted">
                    <i class="fas fa-chart-pie fa-3x mb-3"></i>
                    <h5>No visualizations available</h5>
                    <p>Visualizations are generated for structured numerical data.</p>
                </div>
            `;
            return;
        }

        let html = '';
        for (const [plotType, plotData] of Object.entries(plots)) {
            html += `
                <div class="plot-container mb-4">
                    <h6 class="plot-title">${this.formatTitle(plotType)}</h6>
                    <div id="plot-${plotType}" class="plot-div"></div>
                </div>
            `;
        }

        visualizationsContent.innerHTML = html;

        // Render plots
        setTimeout(() => {
            for (const [plotType, plotData] of Object.entries(plots)) {
                try {
                    // Check if this is a control chart (has chart_image property)
                    if (plotData && typeof plotData === 'object' && plotData.chart_image) {
                        // Handle control chart image
                        const chartContainer = document.getElementById(`plot-${plotType}`);
                        chartContainer.innerHTML = `
                            <div class="control-chart-container">
                                <img src="data:image/png;base64,${plotData.chart_image}" 
                                     class="img-fluid control-chart-image" 
                                     alt="${plotData.type || 'Control Chart'}" />
                                ${plotData.statistics ? this.createControlChartStatsTable(plotData.statistics) : ''}
                            </div>
                        `;
                    } else {
                        // Handle regular Plotly charts
                        const plotJson = JSON.parse(plotData);
                        Plotly.newPlot(`plot-${plotType}`, plotJson.data, plotJson.layout, {
                            responsive: true,
                            displayModeBar: true
                        });
                    }
                } catch (error) {
                    console.error(`Error rendering ${plotType} plot:`, error);
                    document.getElementById(`plot-${plotType}`).innerHTML = `
                        <div class="alert alert-warning">
                            <i class="fas fa-exclamation-triangle me-2"></i>
                            Error rendering ${plotType} visualization
                        </div>
                    `;
                }
            }
        }, 100);
    }

    populateConclusion(conclusion) {
        const conclusionContent = document.getElementById('conclusionContent');
        conclusionContent.innerHTML = `
            <div class="conclusion-text">
                <div class="alert alert-success">
                    <i class="fas fa-flag-checkered me-2"></i>
                    <strong>Conclusion & Recommendations:</strong>
                </div>
                <div class="formatted-text">
                    ${this.formatText(conclusion)}
                </div>
            </div>
        `;
    }

    formatTitle(title) {
        return title.replace(/_/g, ' ')
                   .split(' ')
                   .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                   .join(' ');
    }

    formatResultValue(value) {
        if (typeof value === 'object' && value !== null) {
            if (Array.isArray(value)) {
                return `<ul class="list-unstyled">${value.map(item => `<li>• ${item}</li>`).join('')}</ul>`;
            } else {
                return this.createTable(value);
            }
        } else if (typeof value === 'number') {
            return `<span class="stat-value">${value.toFixed(4)}</span>`;
        } else {
            return `<p>${value}</p>`;
        }
    }

    createTable(data) {
        let html = '<div class="table-responsive"><table class="table table-striped results-table">';
        
        // Handle nested objects (like correlation matrix)
        if (typeof Object.values(data)[0] === 'object') {
            const keys = Object.keys(data);
            const subKeys = Object.keys(Object.values(data)[0]);
            
            // Header
            html += '<thead><tr><th></th>';
            subKeys.forEach(key => {
                html += `<th>${key}</th>`;
            });
            html += '</tr></thead><tbody>';
            
            // Rows
            keys.forEach(key => {
                html += `<tr><th>${key}</th>`;
                subKeys.forEach(subKey => {
                    const value = data[key][subKey];
                    html += `<td>${typeof value === 'number' ? value.toFixed(4) : value}</td>`;
                });
                html += '</tr>';
            });
        } else {
            // Simple key-value pairs
            html += '<thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>';
            Object.entries(data).forEach(([key, value]) => {
                html += `<tr><td>${this.formatTitle(key)}</td><td>${typeof value === 'number' ? value.toFixed(4) : value}</td></tr>`;
            });
        }
        
        html += '</tbody></table></div>';
        return html;
    }

    formatText(text) {
        if (!text) return '';
        
        // Convert line breaks to paragraphs
        return text.split('\n\n')
                  .map(paragraph => `<p>${paragraph.replace(/\n/g, '<br>')}</p>`)
                  .join('');
    }

    async downloadReport() {
        if (!this.currentReportFilename) {
            this.showError('No report available for download');
            return;
        }

        try {
            const response = await fetch(`/download/${this.currentReportFilename}`);
            
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = this.currentReportFilename;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                
                this.showSuccess('Report downloaded successfully!');
            } else {
                throw new Error('Download failed');
            }
        } catch (error) {
            this.showError(`Download failed: ${error.message}`);
        }
    }

    showLoading() {
        document.getElementById('loadingDiv').classList.remove('d-none');
        document.getElementById('resultsCard').classList.add('d-none');
        document.getElementById('errorAlert').classList.add('d-none');
        document.getElementById('analyzeBtn').disabled = true;
        document.getElementById('analyzeBtn').innerHTML = `
            <span class="spinner-border spinner-border-sm me-2" role="status"></span>
            Analyzing...
        `;
    }

    hideLoading() {
        document.getElementById('loadingDiv').classList.add('d-none');
        document.getElementById('analyzeBtn').disabled = false;
        document.getElementById('analyzeBtn').innerHTML = `
            <i class="fas fa-magic me-2"></i>
            Analyze Data
        `;
        this.validateForm(); // Restore button state
    }

    showError(message) {
        const errorAlert = document.getElementById('errorAlert');
        const errorMessage = document.getElementById('errorMessage');
        
        errorMessage.textContent = message;
        errorAlert.classList.remove('d-none');
        errorAlert.scrollIntoView({ behavior: 'smooth' });
        
        // Auto-hide after 10 seconds
        setTimeout(() => {
            errorAlert.classList.add('d-none');
        }, 10000);
    }

    showSuccess(message) {
        // Create temporary success alert
        const successAlert = document.createElement('div');
        successAlert.className = 'alert alert-success fade-in-up';
        successAlert.innerHTML = `
            <i class="fas fa-check-circle me-2"></i>
            ${message}
        `;
        
        // Insert before the upload form
        const uploadForm = document.getElementById('uploadForm').parentElement;
        uploadForm.parentNode.insertBefore(successAlert, uploadForm);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            successAlert.remove();
        }, 5000);
    }

    // Utility methods
    resetForm() {
        document.getElementById('uploadForm').reset();
        this.dropzone.removeAllFiles();
        document.getElementById('resultsCard').classList.add('d-none');
        document.getElementById('downloadBtn').classList.add('d-none');
        this.currentReportFilename = null;
        this.validateForm();
    }

    createControlChartStatsTable(statistics) {
        // Create a compact statistics table for control charts
        const keyStats = ['center_line', 'ucl', 'lcl'];
        let tableHtml = '<div class="control-chart-stats mt-3"><h6>Control Chart Statistics:</h6><table class="table table-sm table-striped">';
        
        for (const stat of keyStats) {
            if (statistics[stat] !== undefined) {
                let statName = stat.replace('_', ' ').split(' ').map(word => 
                    word.charAt(0).toUpperCase() + word.slice(1)
                ).join(' ');
                
                if (stat === 'ucl') statName = 'Upper Control Limit';
                else if (stat === 'lcl') statName = 'Lower Control Limit';
                else if (stat === 'center_line') statName = 'Center Line';
                
                tableHtml += `<tr><td><strong>${statName}:</strong></td><td>${statistics[stat].toFixed(4)}</td></tr>`;
            }
        }
        
        // Add out-of-control points info
        const oocPoints = statistics.out_of_control_points || [];
        if (oocPoints.length > 0) {
            tableHtml += `<tr><td><strong>Out-of-Control Points:</strong></td><td class="text-danger">${oocPoints.length} detected</td></tr>`;
        } else {
            tableHtml += `<tr><td><strong>Process Status:</strong></td><td class="text-success">In Control</td></tr>`;
        }
        
        tableHtml += '</table></div>';
        return tableHtml;
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.autoStatIQ = new AutoStatIQ();
});

// Global utility functions
window.startNewAnalysis = () => {
    window.autoStatIQ.resetForm();
    window.scrollTo({ top: 0, behavior: 'smooth' });
};

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
        // Refresh any real-time elements if needed
        console.log('AutoStatIQ: Page is now visible');
    }
});
