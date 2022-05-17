import base64
import json
from PIL import Image
from io import BytesIO
from flask import Flask, request, Response, jsonify
from test_predict import predict_
from helper_functions import download_model_from_gcs

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

@app.route('/health', methods=['GET'])
def health_check():
    return Response(response=json.dumps({"status": 'healthy'}), status=200, mimetype="application/json")

@app.route('/get_movie_genres', methods=['POST'])
def main():
    
    request_json = request.get_json()
    request_instances = request_json['instances']

    b64_string = request_instances[0]['b64_string']
    predicted_genres = predict_(b64_string)

    output = {'predictions':
               [
                   {
                       'predicted_genres' : predicted_genres
                   }
               ]
           }
    return jsonify(output)

if __name__ == '__main__':
    
    app.run(host='0.0.0.0', port=5000)
