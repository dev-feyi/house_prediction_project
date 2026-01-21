import os
from flask import Flask, render_template, request, jsonify
from model import HousePriceModel

# Gunicorn looks for 'app' in 'app.py' by default
app = Flask(__name__)

# Model Initialization
house_predictor = HousePriceModel()

def bootstrap_model():
    """Ensure the model is loaded or trained before the first request."""
    if not house_predictor.load_model():
        print("Training state not found. Initializing training sequence...")
        from model import train_and_save_model
        train_and_save_model()
        house_predictor.load_model()

bootstrap_model()

def validate_house_data(params):
    """
    Business logic for input validation. 
    Returns (error_message, is_valid)
    """
    ranges = {
        'square_feet': (500, 10000, "Square footage must be between 500 and 10,000"),
        'bedrooms': (1, 10, "Bedrooms must be between 1 and 10"),
        'bathrooms': (1, 8, "Bathrooms must be between 1 and 8"),
        'age_years': (0, 100, "Age must be between 0 and 100 years"),
        'garage_spaces': (0, 4, "Garage spaces must be between 0 and 4"),
        'location_score': (1, 10, "Location score must be between 1 and 10")
    }

    for key, (min_val, max_val, msg) in ranges.items():
        val = float(params.get(key, 0))
        if not (min_val <= val <= max_val):
            return msg, False
    return None, True

@app.route('/')
def index():
    """Serves the front-end application."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def handle_prediction():
    """API endpoint to process house features and return a price estimate."""
    try:
        payload = request.get_json() or {}
        
        # Run validation
        err_msg, is_valid = validate_house_data(payload)
        if not is_valid:
            return jsonify({'success': False, 'error': err_msg}), 400

        # Execute prediction via the model wrapper
        estimate = house_predictor.predict(
            square_feet=float(payload['square_feet']),
            bedrooms=int(payload['bedrooms']),
            bathrooms=float(payload['bathrooms']),
            age_years=int(payload['age_years']),
            garage_spaces=int(payload['garage_spaces']),
            location_score=int(payload['location_score'])
        )

        return jsonify({
            'success': True,
            'predicted_price': round(estimate, 2),
            'metadata': payload  # Echoing back inputs under a different key
        })

    except (KeyError, ValueError) as e:
        return jsonify({'success': False, 'error': f'Invalid input format: {str(e)}'}), 400
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({'success': False, 'error': 'An internal server error occurred'}), 500

if __name__ == '__main__':
    # Configuration for local development
    PORT = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=PORT, debug=True)