from flask import Flask, request, jsonify
import os
#from flask_cors import CORS, cross_origin
from utils.utils import decodeImage
from src.research.obj import MultiClassObj

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
#CORS(app)


#@cross_origin()
class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        modelPath = 'ssd_mobilenet_v1_coco_2017_11_17'
        self.objectDetection = MultiClassObj(self.filename, modelPath)


@app.route("/predict", methods=['POST'])
#@cross_origin()
def predictRoute():
    image = request.json['image']
    decodeImage(image, clApp.filename)
    result = clApp.objectDetection.getPrediction()
    return jsonify(result)


if __name__ == "__main__":
    clApp = ClientApp()
    app.run()
    
