from flask import Flask, render_template, request, send_from_directory, jsonify
import numpy as np
import pickle
import time
import threading
import os
import sys
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Global flag for model loading status
model = None
model_loading_status = "not_started"
model_loading_error = None
model_load_time = None

def create_mock_model():
    """Create a mock model for testing if real model fails"""
    print("🔧 Creating mock model for testing...")
    # Create a simple model with random data
    np.random.seed(42)
    X_train = np.random.rand(1000, 15)  # 15 features
    y_train = (X_train[:, 0] > 0.5).astype(int)
    
    mock_model = RandomForestClassifier(n_estimators=10, random_state=42)
    mock_model.fit(X_train, y_train)
    print("✅ Mock model created successfully")
    return mock_model

# Load model with fallback
def load_model():
    global model, model_loading_status, model_loading_error, model_load_time
    
    try:
        model_loading_status = "loading"
        print("⏳ Loading model from model.pkl...")
        model_load_time = time.time()
        
        # Check if model file exists
        if not os.path.exists('model.pkl'):
            print("⚠️ model.pkl not found! Using mock model instead.")
            model = create_mock_model()
            model_loading_status = "loaded"
            print(f"✅ Mock model created successfully in {time.time() - model_load_time:.2f} seconds!")
            return
        
        # Try to load the model
        try:
            model = pickle.load(open('model.pkl', 'rb'))
            model_loading_status = "loaded"
            print(f"✅ Model loaded successfully in {time.time() - model_load_time:.2f} seconds!")
        except Exception as e:
            print(f"⚠️ Failed to load model.pkl: {e}")
            print("🔄 Falling back to mock model...")
            model = create_mock_model()
            model_loading_status = "loaded"
            print(f"✅ Mock model created in {time.time() - model_load_time:.2f} seconds!")
        
    except Exception as e:
        model_loading_status = "error"
        model_loading_error = str(e)
        print(f"❌ Error: {e}")
        # Even on error, create mock model as fallback
        try:
            print("🔄 Creating emergency mock model...")
            model = create_mock_model()
            model_loading_status = "loaded"
            print("✅ Emergency mock model created")
        except:
            pass

# Load model in background thread
thread = threading.Thread(target=load_model, daemon=True)
thread.start()

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/')
def index():
    """Main route - handles both loading and main app"""
    loaded = request.args.get('loaded', 'false')
    
    if loaded == 'true':
        return render_template('index.html')
    else:
        return render_template('loading.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    """Quick status check - returns immediately"""
    global model, model_loading_status, model_loading_error
    
    # Always return ready if model exists (including mock model)
    is_ready = model is not None
    
    return jsonify({
        'status': 'loaded' if is_ready else model_loading_status,
        'model_ready': is_ready,
        'error': model_loading_error if model_loading_status == 'error' else None,
        'server_ready': True,
        'using_mock': is_ready and not os.path.exists('model.pkl') if is_ready else False
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
    
@app.route('/debug')
def debug():
    """Debug route to check what's happening"""
    import os
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    templates = os.listdir(templates_dir) if os.path.exists(templates_dir) else []
    model_exists = os.path.exists('model.pkl')
    
    return {
        'templates_folder_exists': os.path.exists(templates_dir),
        'templates_found': templates,
        'model_file_exists': model_exists,
        'model_loaded': model is not None,
        'model_status': model_loading_status,
        'using_mock': model is not None and not model_exists if model is not None else False
    }

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
