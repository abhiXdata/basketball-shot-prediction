from flask import Flask, render_template, request, session
import numpy as np
import pickle
import time

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-this-in-production'  # Required for sessions

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    """Check if model is loaded and redirect accordingly with loading UI"""
    # Check if this session has seen the loading screen
    if not session.get('loading_complete', False):
        # First time in this session - show loading screen
        return render_template('loading.html')
    
    # Loading already shown in this session - show main app
    return render_template('index.html')

@app.route('/api/initialize', methods=['GET'])
def initialize():
    """Mark loading as complete and return status"""
    # Mark this session as having completed loading
    session['loading_complete'] = True
    
    # Optional: Add a small delay to make loading screen visible
    # Remove this in production if you don't want the delay
    time.sleep(0.5)
    
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
