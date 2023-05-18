from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/api/predict-video", methods=["POST"])
def video():
    return "REAL"

@app.route("/api/predict-audio", methods=["POST"])
def audio():
    return "FAKE"



if __name__ == '__main__':
    app.run()