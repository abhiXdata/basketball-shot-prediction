from flask import Flask, render_template, request, send_from_directory, jsonify, url_for
import numpy as np
import pickle
import time
import threading
import os
import sys

app = Flask(__name__)

# Global flag for model loading status
model = None
model_loading_status = "not_started"
model_loading_error = None
model_load_time = None

# Load model synchronously but with timeout
def load_model():
    global model, model_loading_status, model_loading_error, model_load_time
    
    try:
        model_loading_status = "loading"
        print("⏳ Loading model from model.pkl...")
        model_load_time = time.time()
        
        # Check if model file exists
        if not os.path.exists('model.pkl'):
            raise FileNotFoundError("model.pkl not found. Please ensure the model file is uploaded.")
        
        # Load the model
        model = pickle.load(open('model.pkl', 'rb'))
        model_loading_status = "loaded"
        print(f"✅ Model loaded successfully in {time.time() - model_load_time:.2f} seconds!")
        
    except Exception as e:
        model_loading_status = "error"
        model_loading_error = str(e)
        print(f"❌ Error loading model: {e}")

# Load model in background thread (non-blocking)
thread = threading.Thread(target=load_model, daemon=True)
thread.start()

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/')
def index():
    """Main route - handles both loading and main app"""
    # Check if this request is coming from loading screen completion
    loaded = request.args.get('loaded', 'false')
    force_loading = request.args.get('loading', 'false')
    
    # If force_loading is true or model not ready and not coming from loaded
    if force_loading == 'true':
        return render_template('loading.html')
    
    # If loaded=true, show main app regardless of model status
    if loaded == 'true':
        # If model is loaded, show main app
        if model is not None:
            return render_template('index.html')
        else:
            # Model not ready yet, show loading with status
            return render_template('loading.html', server_status=model_loading_status)
    
    # First visit - always show loading screen
    return render_template('loading.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    """Quick status check - returns immediately"""
    global model, model_loading_status, model_loading_error
    
    return jsonify({
        'status': model_loading_status,
        'model_ready': model is not None,
        'error': model_loading_error if model_loading_status == 'error' else None,
        'server_ready': True
    })

@app.route('/api/initialize', methods=['GET'])
def initialize():
    """Non-blocking initialization check - returns immediately"""
    global model, model_loading_status
    
    if model is not None:
        return jsonify({
            'status': 'success',
            'message': 'Model ready',
            'ready': True
        })
    elif model_loading_status == 'error':
        return jsonify({
            'status': 'error',
            'message': model_loading_error,
            'ready': False
        })
    else:
        return jsonify({
            'status': 'loading',
            'message': 'Model is still loading',
            'ready': False
        })

@app.route('/predict', methods=['POST'])
def predict():
    global model
    
    if model is None:
        return render_template(
            'index.html',
            prediction_text="⚠️ Model is still loading. Please wait a moment and refresh the page."
        )
    
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
    # Get port from environment variable for Render
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
