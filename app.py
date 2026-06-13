from flask import Flask, render_template, request, send_from_directory, jsonify
import numpy as np
import pickle
import time
import os
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load model immediately (synchronously)
print("=" * 50)
print("🚀 STARTING BASKETBALL SHOT PREDICTOR")
print("=" * 50)

def create_mock_model():
    """Create a mock model for testing"""
    print("🔧 Creating mock model...")
    np.random.seed(42)
    X_train = np.random.rand(1000, 15)
    y_train = (X_train[:, 0] > 0.5).astype(int)
    mock_model = RandomForestClassifier(n_estimators=10, random_state=42)
    mock_model.fit(X_train, y_train)
    print("✅ Mock model created successfully")
    return mock_model

# Try to load the real model, fallback to mock
model = None
model_type = "none"

try:
    print("📂 Checking for model.pkl...")
    if os.path.exists('model.pkl'):
        print("📂 Found model.pkl, loading...")
        model = pickle.load(open('model.pkl', 'rb'))
        model_type = "real"
        print("✅ Real model loaded successfully!")
    else:
        print("⚠️ model.pkl not found!")
        raise FileNotFoundError("model.pkl not found")
except Exception as e:
    print(f"⚠️ Failed to load real model: {e}")
    print("🔄 Creating mock model as fallback...")
    model = create_mock_model()
    model_type = "mock"
    print("✅ Mock model ready!")

print(f"📊 Model type: {model_type}")
print("=" * 50)

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
    """Status check - returns immediately"""
    global model
    
    return jsonify({
        'status': 'loaded',
        'model_ready': model is not None,
        'model_type': model_type,
        'server_ready': True
    })

@app.route('/api/initialize', methods=['GET'])
def initialize():
    """Initialization check - returns immediately"""
    global model
    
    return jsonify({
        'status': 'success',
        'message': 'Model ready',
        'ready': True,
        'model_type': model_type
    })

@app.route('/predict', methods=['POST'])
def predict():
    global model
    
    # Check if it's an AJAX request
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
    
    if model is None:
        if is_ajax:
            return jsonify({'error': 'Model not available'}), 503
        return render_template(
            'index.html',
            prediction_text="⚠️ Model not available. Please refresh the page."
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

        is_made = bool(prediction[0] == 1)
        confidence = round(max(probability[0]) * 100, 2)
        
        result_text = "Shot Made" if is_made else "Shot Missed"
        full_text = f"🏀 {result_text} | Confidence: {confidence}%"
        
        # Return JSON for AJAX requests
        if is_ajax:
            return jsonify({
                'success': True,
                'is_made': is_made,
                'confidence': confidence,
                'result': result_text,
                'full_text': full_text
            })
        
        # Return HTML for regular form submission
        return render_template(
            'index.html',
            prediction_text=full_text
        )

    except Exception as e:
        print("ERROR OCCURRED:", str(e))
        if is_ajax:
            return jsonify({'error': str(e)}), 500
        return render_template(
            'index.html',
            prediction_text=f"Error: {str(e)}"
        )
    
@app.route('/debug')
def debug():
    """Debug route"""
    import os
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    templates = os.listdir(templates_dir) if os.path.exists(templates_dir) else []
    model_exists = os.path.exists('model.pkl')
    
    return {
        'templates_folder_exists': os.path.exists(templates_dir),
        'templates_found': templates,
        'model_file_exists': model_exists,
        'model_loaded': model is not None,
        'model_type': model_type,
        'python_version': os.sys.version
    }

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
