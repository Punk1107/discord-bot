from flask import Flask, jsonify
from threading import Thread
import logging
import socket

# --- Flask app setup ---
app = Flask(__name__)

# Disable Flask’s default logger to avoid duplicate logs in multi-thread
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

@app.route("/")
def home():
    return jsonify({"status": "Bot is running!"})


def find_free_port(default_port=8080, max_tries=10):
    """Try to find an available port starting from default_port."""
    port = default_port
    for _ in range(max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                port += 1
    # fallback: default if all failed
    return default_port


def run_server():
    """Run Flask app safely."""
    try:
        port = find_free_port(8080)
        app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False, threaded=True)
    except Exception as e:
        print(f"[WebServer] Failed to start Flask: {e}")


def start_webserver():
    """Start the Flask webserver on a daemon thread."""
    try:
        thread = Thread(target=run_server, name="FlaskWebServer", daemon=True)
        thread.start()
        print("[WebServer] Flask thread started successfully.")
    except Exception as e:
        print(f"[WebServer] Error starting Flask thread: {e}")
