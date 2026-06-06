from flask import Flask, render_template, request, send_from_directory
import numpy as np
import pickle
import time
import os

app = Flask(__name__)

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

# Global flag to track if loading has been shown
loading_shown = False

@app.route('/favicon.ico')
def favicon():
    """Return empty response for favicon to prevent 404 errors"""
    return '', 204

@app.route('/')
def index():
    """Always show loading screen, then redirect to main app"""
    # Get parameter to check if this is after loading
    loaded = request.args.get('loaded', 'false')
    
    if loaded == 'true':
        # This is after loading screen, show main app
        return render_template('index.html')
    else:
        # First visit or refresh - show loading screen
        return render_template('loading.html')

@app.route('/api/initialize', methods=['GET'])
def initialize():
    """Mark initialization as complete"""
    return {
        'status': 'success',
        'message': 'Loading complete',
        'ready': True
    }

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("FORM DATA:", request.form)

        features = [[
            int(request.form['action_type']),
            int(request.form['combined_shot_type']),
            float(request.form['loc_x']),
            float(request.form['loc_y']),
            int(request.form['period']),
            int(request.form['playoffs']),
            int(request.form['season']),
            int(request.form['shot_distance']),
            int(request.form['shot_type']),
            int(request.form['shot_zone_area']),
            int(request.form['shot_zone_basic']),
            int(request.form['shot_zone_range']),
            int(request.form['matchup']),
            int(request.form['opponent']),
            int(request.form['time_remaining'])
        ]]

        print("FEATURES:", features)

        prediction = model.predict(features)
        probability = model.predict_proba(features)

        result = "🏀 Shot Made" if prediction[0] == 1 else "❌ Shot Missed"
        confidence = round(max(probability[0]) * 100, 2)

        return render_template(
            'index.html',
            prediction_text=f"{result} | Confidence: {confidence}%"
        )

    except Exception as e:
        print("ERROR OCCURRED:", str(e))
        return render_template(
            'index.html',
            prediction_text=f"Error: {str(e)}"
        )
    
if __name__ == "__main__":
    app.run(debug=True)
