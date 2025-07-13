#!/usr/bin/env python3
"""
Production deployment script for the Medical Readmission Prediction Server
Uses Gunicorn for production-ready deployment
"""

import os
import subprocess
import sys

def start_production_server():
    """Start the production server with Gunicorn"""
    
    # Configuration
    host = "0.0.0.0"
    port = os.environ.get('PORT', 8080)
    workers = os.environ.get('WORKERS', 4)
    
    # Gunicorn command
    cmd = [
        'gunicorn',
        '--bind', f'{host}:{port}',
        '--workers', str(workers),
        '--timeout', '120',
        '--preload',
        '--access-logfile', '-',
        '--error-logfile', '-',
        '--log-level', 'info',
        'prediction_server:app'
    ]
    
    print(f"🚀 Starting production server on {host}:{port} with {workers} workers")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Server failed with exit code {e.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    start_production_server()
