from flask import Flask, jsonify, request
from predictions import predict_video_main

app = Flask(__name__)

@app.route("/api/predict-video", methods=["POST"])
def video():
    output = predict_video_main.predict_algorithm()
    return output

@app.route("/api/predict-audio", methods=["POST"])
def audio():
    return "FAKE"



if __name__ == '__main__':
    app.run()