from flask import Flask, render_template, request, send_from_directory, jsonify
import numpy as np
import pickle
import time
import threading
import os

app = Flask(__name__)

# Global flag for model loading status
model = None
model_loading_status = "not_started"  # not_started, loading, loaded, error
model_loading_error = None

# Load model in background to not block startup
def load_model_async():
    global model, model_loading_status, model_loading_error
    try:
        model_loading_status = "loading"
        print("⏳ Loading model in background...")
        time.sleep(1)  # Simulate loading time (remove in production)
        model = pickle.load(open('model.pkl', 'rb'))
        model_loading_status = "loaded"
        print("✅ Model loaded successfully!")
    except Exception as e:
        model_loading_status = "error"
        model_loading_error = str(e)
        print(f"❌ Error loading model: {e}")

# Start loading model in background thread
threading.Thread(target=load_model_async, daemon=True).start()

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/')
def index():
    """Main route - always shows loading first"""
    loaded = request.args.get('loaded', 'false')
    
    if loaded == 'true':
        # Show main app only if model is loaded
        if model is not None:
            return render_template('index.html')
        else:
            # Model not ready yet, show loading with status
            return render_template('loading.html', server_status=model_loading_status)
    else:
        # First load - show loading screen
        return render_template('loading.html', server_status=model_loading_status)

@app.route('/api/status', methods=['GET'])
def get_status():
    """Check server and model initialization status"""
    global model, model_loading_status, model_loading_error
    
    return jsonify({
        'status': model_loading_status,
        'model_ready': model is not None,
        'error': model_loading_error,
        'server_ready': True
    })

@app.route('/api/initialize', methods=['GET'])
def initialize():
    """Wait for model to load and return status"""
    global model, model_loading_status
    
    # Wait for model to load (max 30 seconds)
    timeout = 30
    start_time = time.time()
    
    while model is None and (time.time() - start_time) < timeout:
        if model_loading_status == "error":
            return jsonify({
                'status': 'error',
                'message': model_loading_error,
                'ready': False
            }), 500
        time.sleep(0.5)
    
    if model is not None:
        return jsonify({
            'status': 'success',
            'message': 'Model ready',
            'ready': True
        })
    else:
        return jsonify({
            'status': 'timeout',
            'message': 'Model loading timeout',
            'ready': False
        }), 504

@app.route('/predict', methods=['POST'])
def predict():
    global model
    
    if model is None:
        return render_template(
            'index.html',
            prediction_text="⚠️ Model is still loading. Please wait a moment and try again."
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
    app.run(debug=True, host='0.0.0.0', port=5000)
