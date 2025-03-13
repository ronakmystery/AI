from flask import Flask, jsonify
from classify import classify
from ultralytics import YOLO

app = Flask(__name__)

model = YOLO("yolov8n.pt")  



@app.route('/')
def home():
    return jsonify({"message": "server is running!"})

@app.route('/ai/objects')
def get_data():
    result = classify(model,"test.png")
    print(result)
    
    return jsonify(result)  # Return detected classes and image path


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000,debug=True)
