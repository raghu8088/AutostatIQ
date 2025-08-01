# AutoStatIQ R Client Library
# Ready-to-use R package for AutoStatIQ API integration
# 
# Author: Roman Chaudhary
# Contact: chaudharyroman.com.np
# Version: 1.0.0

# Required packages
if (!require("httr")) install.packages("httr")
if (!require("jsonlite")) install.packages("jsonlite")
if (!require("R6")) install.packages("R6")

library(httr)
library(jsonlite)
library(R6)

#' AutoStatIQ R Client
#' 
#' R6 class for interfacing with the AutoStatIQ API for comprehensive
#' statistical analysis. Supports file uploads, natural language questions,
#' and professional report generation.
#' 
#' @field base_url Base URL of the AutoStatIQ API
#' @field api_key Optional API key for authentication
#' @field timeout Request timeout in seconds
#' 
#' @examples
#' # Initialize client
#' client <- AutoStatIQClient$new()
#' 
#' # Check API health
#' health <- client$check_health()
#' print(health$status)
#' 
#' # Analyze file
#' results <- client$analyze_file("data.csv", "Perform ANOVA analysis")
#' print(results$success)
#' 
#' # Download report
#' client$download_report(results$report_filename, "report.docx")
#' 
#' @export
AutoStatIQClient <- R6Class("AutoStatIQClient",
  public = list(
    base_url = NULL,
    api_key = NULL,
    timeout = NULL,
    
    #' Initialize AutoStatIQ client
    #' 
    #' @param base_url Base URL of AutoStatIQ API (default: http://127.0.0.1:5000)
    #' @param api_key Optional API key for authentication
    #' @param timeout Request timeout in seconds (default: 300)
    initialize = function(base_url = "http://127.0.0.1:5000", api_key = NULL, timeout = 300) {
      self$base_url <- gsub("/$", "", base_url)
      self$api_key <- api_key
      self$timeout <- timeout
    },
    
    #' Check API health and configuration status
    #' 
    #' @return List containing health status and API information
    #' @examples
    #' client <- AutoStatIQClient$new()
    #' health <- client$check_health()
    #' cat("API Status:", health$status, "\n")
    check_health = function() {
      tryCatch({
        response <- self$make_request("/health", "GET")
        return(content(response, "parsed"))
      }, error = function(e) {
        stop(paste("Health check failed:", e$message))
      })
    },
    
    #' Get comprehensive API information and capabilities
    #' 
    #' @return List containing API info, supported methods, and features
    #' @examples
    #' client <- AutoStatIQClient$new()
    #' info <- client$get_api_info()
    #' cat("Statistical Methods:", length(info$statistical_methods), "\n")
    get_api_info = function() {
      tryCatch({
        response <- self$make_request("/api/info", "GET")
        return(content(response, "parsed"))
      }, error = function(e) {
        stop(paste("Failed to get API info:", e$message))
      })
    },
    
    #' Perform statistical analysis on uploaded file
    #' 
    #' @param file_path Path to the file to analyze (CSV, XLSX, DOCX, PDF, JSON, TXT)
    #' @param question Optional natural language statistical question
    #' @return List containing comprehensive analysis results
    #' @examples
    #' client <- AutoStatIQClient$new()
    #' results <- client$analyze_file("data.csv", "Test for significant differences")
    #' if (results$success) {
    #'   cat("Analysis completed successfully!\n")
    #'   cat("Number of plots:", length(results$plots), "\n")
    #' }
    analyze_file = function(file_path, question = NULL) {
      # Validate file exists
      if (!file.exists(file_path)) {
        stop(paste("File not found:", file_path))
      }
      
      # Validate file format
      supported_extensions <- c(".csv", ".xlsx", ".xls", ".docx", ".pdf", ".json", ".txt")
      file_ext <- tools::file_ext(file_path)
      if (!paste0(".", tolower(file_ext)) %in% supported_extensions) {
        stop(paste("Unsupported file format. Supported:", paste(supported_extensions, collapse = ", ")))
      }
      
      tryCatch({
        # Prepare request body
        body <- list(file = upload_file(file_path))
        if (!is.null(question)) {
          body$question <- question
        }
        
        # Make request
        response <- self$make_request("/upload", "POST", body, encode = "multipart")
        return(content(response, "parsed"))
      }, error = function(e) {
        stop(paste("Analysis failed:", e$message))
      })
    },
    
    #' Analyze a natural language statistical question without file upload
    #' 
    #' @param question Statistical question in natural language
    #' @return List containing analysis recommendations and interpretation
    #' @examples
    #' client <- AutoStatIQClient$new()
    #' results <- client$analyze_question("What test should I use for comparing groups?")
    #' cat("Recommendation:", results$results$text_analysis$recommendations, "\n")
    analyze_question = function(question) {
      tryCatch({
        body <- list(question = question)
        response <- self$make_request("/api/analyze", "POST", toJSON(body, auto_unbox = TRUE))
        add_headers("Content-Type" = "application/json")
        return(content(response, "parsed"))
      }, error = function(e) {
        stop(paste("Question analysis failed:", e$message))
      })
    },
    
    #' Download generated statistical analysis report
    #' 
    #' @param report_url_or_id Report URL or report ID
    #' @param save_path Path to save the downloaded report
    #' @return Path to the saved report file
    #' @examples
    #' client <- AutoStatIQClient$new()
    #' # After analysis...
    #' report_path <- client$download_report(results$report_filename, "my_report.docx")
    #' cat("Report saved to:", report_path, "\n")
    download_report = function(report_url_or_id, save_path) {
      # Extract report ID from URL if needed
      if (grepl("/download/", report_url_or_id)) {
        report_id <- basename(report_url_or_id)
      } else if (grepl("^http", report_url_or_id)) {
        report_id <- basename(report_url_or_id)
      } else {
        report_id <- report_url_or_id
      }
      
      tryCatch({
        response <- self$make_request(paste0("/download/", report_id), "GET")
        
        # Save file
        writeBin(content(response, "raw"), save_path)
        return(save_path)
      }, error = function(e) {
        stop(paste("Download failed:", e$message))
      })
    },
    
    #' Analyze multiple files in batch
    #' 
    #' @param file_paths Vector of file paths to analyze
    #' @param questions Optional vector of questions corresponding to each file
    #' @return List of analysis results for each file
    #' @examples
    #' client <- AutoStatIQClient$new()
    #' files <- c("data1.csv", "data2.xlsx")
    #' questions <- c("Descriptive statistics", "Correlation analysis")
    #' results <- client$batch_analyze(files, questions)
    batch_analyze = function(file_paths, questions = NULL) {
      results <- list()
      
      if (is.null(questions)) {
        questions <- rep(NA, length(file_paths))
      }
      
      for (i in seq_along(file_paths)) {
        cat("Analyzing file", i, "of", length(file_paths), ":", file_paths[i], "\n")
        
        tryCatch({
          result <- self$analyze_file(file_paths[i], questions[i])
          results[[i]] <- result
          
          # Add delay between requests
          if (i < length(file_paths)) {
            Sys.sleep(1)
          }
        }, error = function(e) {
          results[[i]] <- list(
            success = FALSE,
            error = e$message,
            file = file_paths[i]
          )
        })
      }
      
      return(results)
    },
    
    #' Make HTTP request with proper headers and error handling
    #' 
    #' @param endpoint API endpoint
    #' @param method HTTP method
    #' @param body Request body
    #' @param encode Encoding method
    #' @return HTTP response
    make_request = function(endpoint, method = "GET", body = NULL, encode = "json") {
      url <- paste0(self$base_url, endpoint)
      
      # Prepare headers
      headers <- list(
        "User-Agent" = "AutoStatIQ-R-Client/1.0.0"
      )
      
      if (!is.null(self$api_key)) {
        headers[["X-API-Key"]] <- self$api_key
      }
      
      # Make request based on method
      if (method == "GET") {
        response <- GET(url, add_headers(.headers = headers), timeout(self$timeout))
      } else if (method == "POST") {
        if (encode == "multipart") {
          response <- POST(url, body = body, encode = encode, add_headers(.headers = headers), timeout(self$timeout))
        } else {
          headers[["Content-Type"]] <- "application/json"
          response <- POST(url, body = body, add_headers(.headers = headers), timeout(self$timeout))
        }
      }
      
      # Check for HTTP errors
      if (http_error(response)) {
        error_content <- content(response, "text")
        stop(paste("HTTP", status_code(response), ":", error_content))
      }
      
      return(response)
    },
    
    #' Print client information
    print = function() {
      cat("AutoStatIQClient:\n")
      cat("  Base URL:", self$base_url, "\n")
      cat("  API Key:", if(is.null(self$api_key)) "Not set" else "Set", "\n")
      cat("  Timeout:", self$timeout, "seconds\n")
    }
  )
)

# Convenience functions for quick usage

#' Quick one-line analysis function
#' 
#' @param file_path Path to file to analyze
#' @param question Optional statistical question
#' @param base_url API base URL
#' @return Analysis results list
#' @examples
#' results <- quick_analysis("data.csv", "Perform t-test")
#' @export
quick_analysis <- function(file_path, question = NULL, base_url = "http://127.0.0.1:5000") {
  client <- AutoStatIQClient$new(base_url)
  return(client$analyze_file(file_path, question))
}

#' Analyze multiple files in batch
#' 
#' @param file_paths Vector of file paths to analyze
#' @param questions Optional vector of questions for each file
#' @param base_url API base URL
#' @return List of analysis results
#' @examples
#' files <- c("data1.csv", "data2.xlsx")
#' results <- batch_analysis(files)
#' @export
batch_analysis <- function(file_paths, questions = NULL, base_url = "http://127.0.0.1:5000") {
  client <- AutoStatIQClient$new(base_url)
  return(client$batch_analyze(file_paths, questions))
}

#' Quick check if AutoStatIQ API is available and healthy
#' 
#' @param base_url API base URL
#' @return Logical indicating if API is healthy
#' @examples
#' if (check_api_status()) {
#'   cat("API is available!\n")
#' }
#' @export
check_api_status <- function(base_url = "http://127.0.0.1:5000") {
  tryCatch({
    client <- AutoStatIQClient$new(base_url)
    health <- client$check_health()
    return(health$status == "healthy")
  }, error = function(e) {
    return(FALSE)
  })
}

# Package metadata and examples
if (interactive()) {
  cat("AutoStatIQ R Client loaded successfully!\n")
  cat("===========================================\n")
  cat("Example usage:\n")
  cat("1. client <- AutoStatIQClient$new()\n")
  cat("2. health <- client$check_health()\n")
  cat("3. results <- client$analyze_file('data.csv', 'Perform ANOVA')\n")
  cat("4. client$download_report(results$report_filename, 'report.docx')\n")
  cat("\n")
  cat("Quick functions:\n")
  cat("- quick_analysis('data.csv', 'Your question')\n")
  cat("- batch_analysis(c('file1.csv', 'file2.xlsx'))\n")
  cat("- check_api_status()\n")
  cat("\n")
  cat("Supported file formats: CSV, XLSX, DOCX, PDF, JSON, TXT\n")
  cat("Statistical methods: 25+ including descriptive, inferential, and advanced analytics\n")
  cat("Features: Control charts, PCA, clustering, hypothesis testing, and more!\n")
}

# Example analysis function for demonstration
demo_analysis <- function() {
  cat("AutoStatIQ R Client Demo\n")
  cat("========================\n")
  
  # Initialize client
  client <- AutoStatIQClient$new()
  
  # Check API health
  tryCatch({
    health <- client$check_health()
    cat("✅ API Status:", health$status, "\n")
    cat("📊 Version:", health$version, "\n")
    cat("🔧 Features:", paste(health$features, collapse = ", "), "\n")
    cat("\n")
  }, error = function(e) {
    cat("❌ API not available:", e$message, "\n")
    cat("Please ensure AutoStatIQ server is running on http://127.0.0.1:5000\n")
    return()
  })
  
  # Get API information
  tryCatch({
    api_info <- client$get_api_info()
    cat("📈 Statistical Methods:", length(api_info$statistical_methods), "\n")
    cat("📁 Supported Formats:", paste(api_info$supported_formats, collapse = ", "), "\n")
    cat("📊 Control Charts:", length(api_info$control_chart_types), "\n")
    cat("\n")
  }, error = function(e) {
    cat("⚠️ Could not get API info:", e$message, "\n")
  })
  
  cat("To use this client with your data:\n")
  cat("1. client <- AutoStatIQClient$new()\n")
  cat("2. results <- client$analyze_file('your_data.csv', 'your_question')\n")
  cat("3. client$download_report(results$report_filename, 'report.docx')\n")
  cat("\n")
  
  cat("For natural language analysis:\n")
  cat("client$analyze_question('What test should I use for comparing two groups?')\n")
  cat("\n")
  
  cat("For batch processing:\n")
  cat("results <- client$batch_analyze(c('file1.csv', 'file2.xlsx'), c('question1', 'question2'))\n")
  cat("\n")
  
  cat("💡 This client supports all AutoStatIQ features:\n")
  cat("   • 25+ Statistical Methods\n")
  cat("   • Control Charts (SPC)\n")
  cat("   • Advanced Analytics (PCA, Clustering)\n")
  cat("   • Professional Report Generation\n")
  cat("   • Intelligent Interpretations\n")
  cat("   • Multi-format File Support\n")
}

# Export main class and functions
if (exists("export")) {
  export(AutoStatIQClient)
  export(quick_analysis)
  export(batch_analysis) 
  export(check_api_status)
  export(demo_analysis)
}
