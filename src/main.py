#!/usr/bin/env python3
"""
Premium AI Signals Backend - Main Entry Point
Production deployment entry point for the Flask application with SocketIO support.
"""

import sys
import os

# Add the parent directory to the Python path to import the app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from advanced_enhanced_app import app, socketio

if __name__ == '__main__':
    # For production deployment
    port = int(os.environ.get('PORT', 8000))
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)

