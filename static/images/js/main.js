// Update value displays for range inputs
document.getElementById('distance').addEventListener('input', function(e) {
    document.getElementById('distanceValue').textContent = e.target.value + ' ft';
});

document.getElementById('angle').addEventListener('input', function(e) {
    document.getElementById('angleValue').textContent = e.target.value + '°';
});

document.getElementById('defenderDistance').addEventListener('input', function(e) {
    document.getElementById('defenderDistanceValue').textContent = e.target.value + ' ft';
});

document.getElementById('shotClock').addEventListener('input', function(e) {
    document.getElementById('shotClockValue').textContent = e.target.value + ' s';
});

// Check if model is ready when page loads
document.addEventListener('DOMContentLoaded', async function() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        
        if (!data.model_loaded) {
            console.warn('Model not loaded. Please refresh the page.');
            showNotification('Model is still loading. Please wait...', 'warning');
        }
    } catch (error) {
        console.error('Health check failed:', error);
    }
});

// Handle prediction
document.getElementById('predictBtn').addEventListener('click', async function() {
    const button = this;
    const resultCard = document.getElementById('result');
    
    // Show loading state
    button.classList.add('loading');
    button.textContent = 'Predicting...';
    
    // Collect input values
    const shotData = {
        distance: parseFloat(document.getElementById('distance').value),
        angle: parseFloat(document.getElementById('angle').value),
        defender_distance: parseFloat(document.getElementById('defenderDistance').value),
        shot_clock: parseFloat(document.getElementById('shotClock').value),
        is_open: parseInt(document.getElementById('isOpen').value)
    };
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(shotData)
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            displayResult(result);
            resultCard.style.display = 'block';
            
            // Scroll to result
            resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        } else {
            showNotification('Error: ' + result.message, 'error');
        }
    } catch (error) {
        console.error('Prediction error:', error);
        showNotification('Failed to get prediction. Please try again.', 'error');
    } finally {
        // Remove loading state
        button.classList.remove('loading');
        button.textContent = 'Predict Shot';
    }
});

function displayResult(result) {
    const predictionText = document.getElementById('predictionText');
    const confidenceText = document.getElementById('confidenceText');
    const ball = document.getElementById('ball');
    
    // Remove previous animation classes
    ball.classList.remove('make-animation', 'miss-animation');
    
    // Force reflow to restart animation
    void ball.offsetWidth;
    
    if (result.prediction === 1) {
        predictionText.textContent = '🏀 SWISH! The shot is predicted to GO IN! 🏀';
        predictionText.className = 'prediction-text make';
        confidenceText.textContent = `Confidence: ${(result.probability * 100).toFixed(1)}%`;
        ball.classList.add('make-animation');
    } else {
        predictionText.textContent = '💔 CLANK! The shot is predicted to MISS! 💔';
        predictionText.className = 'prediction-text miss';
        confidenceText.textContent = `Confidence: ${(result.probability * 100).toFixed(1)}%`;
        ball.classList.add('miss-animation');
    }
}

function showNotification(message, type) {
    // Create notification element
    const notification = document.createElement('div');
    notification.textContent = message;
    notification.style.position = 'fixed';
    notification.style.top = '20px';
    notification.style.right = '20px';
    notification.style.padding = '15px 20px';
    notification.style.backgroundColor = type === 'error' ? '#f44336' : '#FF6B35';
    notification.style.color = 'white';
    notification.style.borderRadius = '10px';
    notification.style.zIndex = '1000';
    notification.style.animation = 'slideIn 0.3s ease-out';
    
    document.body.appendChild(notification);
    
    // Remove after 3 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-out';
        setTimeout(() => {
            notification.remove();
        }, 300);
    }, 3000);
}

// Add CSS animations for notifications
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Optional: Add keyboard shortcut (Enter key for prediction)
document.addEventListener('keypress', function(e) {
    if (e.key === 'Enter' && !document.getElementById('predictBtn').classList.contains('loading')) {
        document.getElementById('predictBtn').click();
    }
});
