#!/usr/bin/env python3
import os
import time
import logging
import datetime
import psutil
import threading
from flask import Flask, request, jsonify
from werkzeug.middleware.proxy_fix import ProxyFix
import logging.config

# Configure logging
log_directory = "logs"
os.makedirs(log_directory, exist_ok=True)

# Create a custom logger
logger = logging.getLogger("dos_test_server")
logger.setLevel(logging.INFO)

# Create handlers
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = os.path.join(log_directory, f"server_{timestamp}.log")

# File handler for all logs
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# Console handler for INFO and above
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatters and add them to handlers
log_format = "%(asctime)s - %(levelname)s - %(message)s"
file_formatter = logging.Formatter(log_format)
file_handler.setFormatter(file_formatter)
console_formatter = logging.Formatter(log_format)
console_handler.setFormatter(console_formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Disable Flask's default logging to console
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.setLevel(logging.ERROR)  # Only show errors
werkzeug_logger.addHandler(file_handler)  # Add file handler to keep logs in file

# Create request logger (disabled by default)
request_logger = logging.getLogger("request_logger")
request_logger.setLevel(logging.CRITICAL)  # Set to CRITICAL to disable INFO logs

# Create a Flask application
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_host=1)
app.logger.setLevel(logging.ERROR)  # Only show errors from Flask logger

# Global metrics
request_count = 0
active_connections = 0
start_time = time.time()
metrics_lock = threading.Lock()

# Metrics tracking class
class ServerMetrics:
    def __init__(self):
        self.total_requests = 0
        self.active_connections = 0
        self.request_times = []  # For tracking response times
        self.requests_per_minute = {}
        self.errors = 0

    def add_request(self, duration):
        self.total_requests += 1
        self.request_times.append(duration)
        # Keep only last 1000 request times
        if len(self.request_times) > 1000:
            self.request_times.pop(0)
        
        # Track requests per minute
        minute = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        if minute not in self.requests_per_minute:
            self.requests_per_minute[minute] = 0
        self.requests_per_minute[minute] += 1
        
        # Clean up old minutes
        current_time = datetime.datetime.now()
        keys_to_remove = []
        for key in self.requests_per_minute:
            key_time = datetime.datetime.strptime(key, "%Y-%m-%d %H:%M")
            if (current_time - key_time).total_seconds() > 3600:  # Remove data older than 1 hour
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del self.requests_per_minute[key]

    def add_connection(self):
        self.active_connections += 1

    def remove_connection(self):
        self.active_connections = max(0, self.active_connections - 1)

    def add_error(self):
        self.errors += 1

    def get_avg_response_time(self):
        if not self.request_times:
            return 0
        return sum(self.request_times) / len(self.request_times)

    def get_requests_per_minute(self):
        current_minute = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        return self.requests_per_minute.get(current_minute, 0)

    def get_metrics(self):
        return {
            "total_requests": self.total_requests,
            "active_connections": self.active_connections,
            "avg_response_time": self.get_avg_response_time(),
            "requests_per_minute": self.get_requests_per_minute(),
            "errors": self.errors
        }

# Initialize metrics
metrics = ServerMetrics()

# Resource monitoring thread
def monitor_resources():
    while True:
        try:
            process = psutil.Process(os.getpid())
            cpu_percent = process.cpu_percent(interval=1)
            memory_info = process.memory_info()
            
            logger.info(f"Resource usage - CPU: {cpu_percent}% | Memory: {memory_info.rss / 1024 / 1024:.2f} MB")
            
            # Log current metrics
            with metrics_lock:
                current_metrics = metrics.get_metrics()
                
            logger.info(
                f"Server metrics - "
                f"Requests: {current_metrics['total_requests']} | "
                f"Active: {current_metrics['active_connections']} | "
                f"Avg Response: {current_metrics['avg_response_time']:.4f}s | "
                f"Rate: {current_metrics['requests_per_minute']}/min | "
                f"Errors: {current_metrics['errors']}"
            )
            
            time.sleep(1)  # Log every 5 seconds
        except Exception as e:
            logger.error(f"Error in monitoring thread: {str(e)}")
            time.sleep(5)

# Start the monitoring thread
monitoring_thread = threading.Thread(target=monitor_resources, daemon=True)
monitoring_thread.start()

@app.before_request
def before_request():
    request.start_time = time.time()
    with metrics_lock:
        metrics.add_connection()
    
    # Log the request (now disabled)
    client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
    request_logger.info(f"Request from {client_ip} - {request.method} {request.path}")

@app.after_request
def after_request(response):
    request_duration = time.time() - request.start_time
    with metrics_lock:
        metrics.remove_connection()
        metrics.add_request(request_duration)
        
        if response.status_code >= 400:
            metrics.add_error()
    
    # Log the response (now disabled)
    request_logger.info(f"Response: {response.status_code} - Took: {request_duration:.4f}s")
    return response

@app.route('/')
def index():
    return 'DoS Test Server is running!'

@app.route('/echo', methods=['POST'])
def echo():
    # Simulate some processing time
    time.sleep(0.05)
    data = request.get_json(silent=True) or {}
    return jsonify(data)

@app.route('/status')
def status():
    uptime = time.time() - start_time
    with metrics_lock:
        current_metrics = metrics.get_metrics()
    
    process = psutil.Process(os.getpid())
    cpu_percent = process.cpu_percent(interval=0.1)
    memory_info = process.memory_info()
    
    return jsonify({
        "status": "running",
        "uptime": uptime,
        "metrics": current_metrics,
        "resources": {
            "cpu_percent": cpu_percent,
            "memory_mb": memory_info.rss / 1024 / 1024
        }
    })

@app.route('/heavy', methods=['GET'])
def heavy_operation():
    """Endpoint that simulates a CPU-intensive operation"""
    # Simulate CPU-intensive task
    duration = int(request.args.get('duration', 1))
    if duration > 10:
        duration = 10  # Cap at 10 seconds for safety
        
    start = time.time()
    while time.time() - start < duration:
        # Perform computation to consume CPU
        [i**2 for i in range(10000)]
    
    return jsonify({"message": f"Heavy operation completed in {time.time() - start:.2f} seconds"})

@app.route('/slow', methods=['GET'])
def slow_response():
    """Endpoint that simulates a slow response"""
    delay = float(request.args.get('delay', 1))
    if delay > 10:
        delay = 10  # Cap at 10 seconds for safety
    
    time.sleep(delay)
    return jsonify({"message": f"Response delayed by {delay} seconds"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting DoS Test Server on port {port}")
    logger.info(f"Log file location: {log_file}")
    logger.info(f"Request logging is DISABLED in console but saved to log file")
    
    # Use threaded=True for better concurrency
    app.run(host='0.0.0.0', port=port, threaded=True) 