from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle
import joblib

app = Flask(__name__)

# Load preprocessor and models (Make sure filenames are correct!)
with open('rice_yield_predictor.pkl', 'rb') as f:
    rice_model = pickle.load(f)
with open('wheat_yield_predictor.pkl', 'rb') as f:
    wheat_model = pickle.load(f)

with open('standard_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('crop_recommendation_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('target_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
loaded_model = joblib.load('crop_recommendation_model.pkl')
print(type(loaded_model))


@app.route('/', methods=['GET'])
def index():
    return render_template('home.html')

@app.route('/yield-prediction', methods=['GET'])
def yield_prediction():
    return render_template('yield-prediction.html')

@app.route('/crop-recommendation', methods=['GET'])
def crop_recommendation():
    return render_template('crop-recommendation.html')

@app.route('/predict_yield', methods=['POST'])
def predict_yield():
    try:
        data = request.get_json()
        print(f"Received data: {data}") #Debugging statement

        crop_type = data['cropType']
        features = {
            'Crop_Year': data['cropYear'],
            'Season': data['season'],
            'State': data['state'],
            'Area': data['area'],
            'Annual_Rainfall': data['annualRainfall'],
            'Fertilizer': data['fertilizer'],
            'Pesticide': data['pesticide']
        }

        features_df = pd.DataFrame([features], columns=features.keys())

        if crop_type == 'rice':
            prediction = rice_model.predict(features_df)[0]
        elif crop_type == 'wheat':
            prediction = wheat_model.predict(features_df)[0]
        else:
            return jsonify({'error': 'Invalid crop type'}), 400

        return jsonify({'prediction': prediction})
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    try:
        data = request.get_json()
        features = {
            'Nitrogen': float(data['nitrogen']),
            'Phosphorus': float(data['phosphorus']),
            'Potassium': float(data['potassium']),
            'Temperature': float(data['temperature']),
            'Humidity': float(data['humidity']),
            'pH_Value': float(data['pH']),
            'Rainfall': float(data['rainfall'])
        }
        
        # Convert features to DataFrame (important to ensure column names are used)
        features_df = pd.DataFrame([features])
    
        
        # Make prediction using the model
        prediction = loaded_model.predict(features_df)[0]
        
        # Decode the prediction if necessary (assuming the model outputs labels)
        decoded_prediction = label_encoder.inverse_transform([prediction])[0]
        
        return jsonify({'recommendation': decoded_prediction})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port = 8080)