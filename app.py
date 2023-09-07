from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import io
import os
import cv2
from PIL import Image
from model import get_predicted_roll_numbers
import base64

app = Flask(__name__)
CORS(app)


knn_model = joblib.load(r'pre_trained_models\KNN_model.joblib')
logistic_regression_model = joblib.load(r'pre_trained_models\KNN_model.joblib')
random_forest_model = joblib.load(r'pre_trained_models\RandomForest_model.joblib')

# knn_model = joblib.load(r'C:\Users\Venkatesh Yeturi\Desktop\auto_attendance_python\Data\knn_model_for_data_1.joblib')

@app.route('/predict_roll_numbers',methods=['POST'])
def predict_roll_numbers_api():

    if 'images' not in request.files:
        return jsonify({'error': 'No images provided'})

    # Get the image file from the form data
    image_files = request.files.getlist('images')
    final_list_of_roll_numbers = set()
    i = 0
    for image_file in image_files:
        print(i)
        i+=1
        image_data = Image.open(io.BytesIO(image_file.read()))
        image_data = Image.open(image_file)
        
        roll_numbers = get_predicted_roll_numbers(knn_model, image_data)

        formatted_roll_numbers = [roll_number.tolist()for roll_number in roll_numbers]
        formatted_roll_numbers = [roll_number[0][:9] for roll_number in formatted_roll_numbers]

        for roll_number in formatted_roll_numbers : 
            final_list_of_roll_numbers.add(roll_number)
    


    return jsonify({'roll_numbers':list(final_list_of_roll_numbers)})

 
if __name__ == '__main__':
    app.run(debug = True)


