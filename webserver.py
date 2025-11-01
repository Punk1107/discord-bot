from flask import Flask, jsonify
from threading import Thread

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"status": "Bot is running!"})

def run():
    # debug=False, use_reloader=False important when running in thread
    app.run(host="0.0.0.0", port=8080, debug=False, use_reloader=False)

def start_webserver():
    # Start Flask in a separate daemon thread so it won't block main bot
    thread = Thread(target=run)
    thread.daemon = True
    thread.start()