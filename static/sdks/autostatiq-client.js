/**
 * AutoStatIQ JavaScript Client Library
 * Ready-to-use JavaScript SDK for AutoStatIQ API integration
 * 
 * Author: Roman Chaudhary
 * Contact: chaudharyroman.com.np
 * Version: 1.0.0
 * 
 * Compatible with both Browser and Node.js environments
 */

// Check if we're in Node.js environment
const isNode = typeof window === 'undefined';

// Import fetch if in Node.js
let fetch;
let FormData;

if (isNode) {
    // Node.js environment
    try {
        fetch = require('node-fetch').default || require('node-fetch');
        FormData = require('form-data');
    } catch (e) {
        console.warn('node-fetch not found. Please install: npm install node-fetch form-data');
    }
} else {
    // Browser environment
    fetch = window.fetch;
    FormData = window.FormData;
}

/**
 * AutoStatIQ JavaScript Client
 * 
 * Provides a simple interface to interact with the AutoStatIQ API
 * for comprehensive statistical analysis.
 */
class AutoStatIQClient {
    /**
     * Initialize AutoStatIQ client
     * 
     * @param {string} baseUrl - Base URL of AutoStatIQ API
     * @param {string|null} apiKey - Optional API key for authentication
     * @param {number} timeout - Request timeout in milliseconds
     */
    constructor(baseUrl = 'http://127.0.0.1:5000', apiKey = null, timeout = 300000) {
        this.baseUrl = baseUrl.replace(/\/$/, '');
        this.timeout = timeout;
        this.headers = {
            'User-Agent': 'AutoStatIQ-JavaScript-Client/1.0.0'
        };
        
        if (apiKey) {
            this.headers['X-API-Key'] = apiKey;
        }
    }

    /**
     * Check API health and configuration status
     * 
     * @returns {Promise<Object>} Health status and API information
     * @throws {Error} If health check fails
     */
    async checkHealth() {
        try {
            const response = await this._makeRequest('/health', 'GET');
            return await response.json();
        } catch (error) {
            throw new Error(`Health check failed: ${error.message}`);
        }
    }

    /**
     * Get comprehensive API information and capabilities
     * 
     * @returns {Promise<Object>} API info, supported methods, and features
     * @throws {Error} If API info request fails
     */
    async getApiInfo() {
        try {
            const response = await this._makeRequest('/api/info', 'GET');
            return await response.json();
        } catch (error) {
            throw new Error(`Failed to get API info: ${error.message}`);
        }
    }

    /**
     * Perform statistical analysis on uploaded file
     * 
     * @param {File|string} file - File object (browser) or file path (Node.js)
     * @param {string|null} question - Optional natural language statistical question
     * @returns {Promise<Object>} Comprehensive analysis results
     * @throws {Error} If analysis fails
     */
    async analyzeFile(file, question = null) {
        try {
            const formData = new FormData();
            
            if (isNode) {
                // Node.js environment - file should be a path
                const fs = require('fs');
                const path = require('path');
                
                if (typeof file !== 'string') {
                    throw new Error('In Node.js, file parameter should be a file path string');
                }
                
                if (!fs.existsSync(file)) {
                    throw new Error(`File not found: ${file}`);
                }
                
                const fileName = path.basename(file);
                const fileStream = fs.createReadStream(file);
                formData.append('file', fileStream, fileName);
            } else {
                // Browser environment - file should be a File object
                if (!(file instanceof File)) {
                    throw new Error('In browser, file parameter should be a File object');
                }
                
                formData.append('file', file);
            }
            
            if (question) {
                formData.append('question', question);
            }

            const response = await this._makeRequest('/upload', 'POST', formData);
            return await response.json();
        } catch (error) {
            throw new Error(`Analysis failed: ${error.message}`);
        }
    }

    /**
     * Analyze a natural language statistical question without file upload
     * 
     * @param {string} question - Statistical question in natural language
     * @returns {Promise<Object>} Analysis recommendations and interpretation
     * @throws {Error} If analysis fails
     */
    async analyzeQuestion(question) {
        try {
            const response = await this._makeRequest('/api/analyze', 'POST', 
                JSON.stringify({ question }), 
                { 'Content-Type': 'application/json' }
            );
            return await response.json();
        } catch (error) {
            throw new Error(`Question analysis failed: ${error.message}`);
        }
    }

    /**
     * Download generated statistical analysis report
     * 
     * @param {string} reportUrlOrId - Report URL or report ID
     * @param {string} filename - Filename to save as (browser only)
     * @returns {Promise<Blob|Buffer>} Report file data
     * @throws {Error} If download fails
     */
    async downloadReport(reportUrlOrId, filename = 'report.docx') {
        try {
            // Extract report ID from URL if needed
            let reportId = reportUrlOrId;
            if (reportUrlOrId.includes('/download/')) {
                reportId = reportUrlOrId.split('/').pop();
            } else if (reportUrlOrId.startsWith('http')) {
                reportId = reportUrlOrId.split('/').pop();
            }

            const response = await this._makeRequest(`/download/${reportId}`, 'GET');
            
            if (isNode) {
                // Node.js environment - return buffer
                return await response.buffer();
            } else {
                // Browser environment - trigger download
                const blob = await response.blob();
                this._triggerDownload(blob, filename);
                return blob;
            }
        } catch (error) {
            throw new Error(`Download failed: ${error.message}`);
        }
    }

    /**
     * Analyze multiple files in batch
     * 
     * @param {Array} files - Array of files or file paths
     * @param {Array|null} questions - Optional array of questions for each file
     * @returns {Promise<Array>} Array of analysis results
     */
    async batchAnalyze(files, questions = null) {
        const results = [];
        const questionsArray = questions || new Array(files.length).fill(null);
        
        for (let i = 0; i < files.length; i++) {
            try {
                console.log(`Analyzing file ${i + 1}/${files.length}...`);
                const result = await this.analyzeFile(files[i], questionsArray[i]);
                results.push(result);
                
                // Add small delay between requests
                if (i < files.length - 1) {
                    await this._delay(1000);
                }
            } catch (error) {
                results.push({
                    success: false,
                    error: error.message,
                    file: files[i]
                });
            }
        }
        
        return results;
    }

    /**
     * Make HTTP request with proper headers and error handling
     * 
     * @private
     * @param {string} endpoint - API endpoint
     * @param {string} method - HTTP method
     * @param {*} body - Request body
     * @param {Object} additionalHeaders - Additional headers
     * @returns {Promise<Response>} Fetch response
     */
    async _makeRequest(endpoint, method = 'GET', body = null, additionalHeaders = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const headers = { ...this.headers, ...additionalHeaders };
        
        const config = {
            method,
            headers,
            timeout: this.timeout
        };
        
        if (body) {
            config.body = body;
        }
        
        const response = await fetch(url, config);
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP ${response.status}: ${errorText}`);
        }
        
        return response;
    }

    /**
     * Trigger file download in browser
     * 
     * @private
     * @param {Blob} blob - File blob
     * @param {string} filename - Filename
     */
    _triggerDownload(blob, filename) {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    }

    /**
     * Simple delay utility
     * 
     * @private
     * @param {number} ms - Milliseconds to delay
     * @returns {Promise} Delay promise
     */
    _delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    toString() {
        return `AutoStatIQClient(baseUrl='${this.baseUrl}')`;
    }
}

// Convenience functions for quick usage

/**
 * Quick one-line analysis function
 * 
 * @param {File|string} file - File to analyze
 * @param {string} question - Optional statistical question
 * @param {string} baseUrl - API base URL
 * @returns {Promise<Object>} Analysis results
 */
async function quickAnalysis(file, question = null, baseUrl = 'http://127.0.0.1:5000') {
    const client = new AutoStatIQClient(baseUrl);
    return await client.analyzeFile(file, question);
}

/**
 * Analyze multiple files in batch
 * 
 * @param {Array} files - Array of files to analyze
 * @param {Array} questions - Optional array of questions
 * @param {string} baseUrl - API base URL
 * @returns {Promise<Array>} Array of analysis results
 */
async function batchAnalysis(files, questions = null, baseUrl = 'http://127.0.0.1:5000') {
    const client = new AutoStatIQClient(baseUrl);
    return await client.batchAnalyze(files, questions);
}

/**
 * Quick check if AutoStatIQ API is available and healthy
 * 
 * @param {string} baseUrl - API base URL
 * @returns {Promise<boolean>} True if API is healthy
 */
async function checkApiStatus(baseUrl = 'http://127.0.0.1:5000') {
    try {
        const client = new AutoStatIQClient(baseUrl);
        const health = await client.checkHealth();
        return health.status === 'healthy';
    } catch (error) {
        return false;
    }
}

// Export for different environments
if (isNode) {
    // Node.js exports
    module.exports = {
        AutoStatIQClient,
        quickAnalysis,
        batchAnalysis,
        checkApiStatus
    };
} else {
    // Browser globals
    window.AutoStatIQClient = AutoStatIQClient;
    window.quickAnalysis = quickAnalysis;
    window.batchAnalysis = batchAnalysis;
    window.checkApiStatus = checkApiStatus;
}

// Example usage documentation
const USAGE_EXAMPLES = {
    browser: `
// Browser Usage Example
const client = new AutoStatIQClient();

// Check API health
const health = await client.checkHealth();
console.log('API Status:', health.status);

// Analyze file from file input
const fileInput = document.getElementById('fileInput');
fileInput.addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (file) {
        try {
            const results = await client.analyzeFile(
                file, 
                'Perform correlation analysis and hypothesis testing'
            );
            
            console.log('Analysis Results:', results);
            
            if (results.success) {
                // Download report
                await client.downloadReport(results.report_filename);
            }
        } catch (error) {
            console.error('Analysis failed:', error);
        }
    }
});
    `,
    nodejs: `
// Node.js Usage Example
const { AutoStatIQClient } = require('./autostatiq-client');

async function analyzeData() {
    const client = new AutoStatIQClient();
    
    try {
        // Check API health
        const health = await client.checkHealth();
        console.log('API Status:', health.status);
        
        // Analyze file
        const results = await client.analyzeFile(
            'data.csv',
            'Test for normality and perform outlier detection'
        );
        
        console.log('Analysis Success:', results.success);
        
        if (results.success) {
            // Save report
            const reportBuffer = await client.downloadReport(results.report_filename);
            const fs = require('fs');
            fs.writeFileSync('statistical_report.docx', reportBuffer);
            console.log('Report saved successfully');
        }
    } catch (error) {
        console.error('Error:', error.message);
    }
}

analyzeData();
    `
};

// Log usage examples if in development
if (typeof process !== 'undefined' && process.env.NODE_ENV === 'development') {
    console.log('AutoStatIQ JavaScript Client loaded successfully!');
    console.log('Browser usage example:', USAGE_EXAMPLES.browser);
    console.log('Node.js usage example:', USAGE_EXAMPLES.nodejs);
}
