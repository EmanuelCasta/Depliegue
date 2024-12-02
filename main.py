from flask import Flask, request, jsonify
from catboost import CatBoostClassifier
import numpy as np
from flask_swagger_ui import get_swaggerui_blueprint


app = Flask(__name__)

model = CatBoostClassifier()
model.load_model("catboost_model.cbm")

SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'
swaggerui_blueprint = get_swaggerui_blueprint(SWAGGER_URL, API_URL, config={'app_name': "CatBoost Prediction API"})
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

PASSWORD = "ProyectoIntegrador"


@app.before_request
def authenticate():
    if request.endpoint == 'predict':
        auth = request.headers.get("Authorization")
        if not auth or auth != f"Bearer {PASSWORD}":
            return jsonify({"error": "Unauthorized"}), 401

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    required_features = ['h_std', 'h_sum', 't_std', 't_sum', 
                         'pr_promedio', 'pr_std', 'pr_sum', 
                         'vv_sum', 'p_sum']
    
    missing_features = [feature for feature in required_features if feature not in data]
    if missing_features:
        return jsonify({
            "error": "Missing required features",
            "missing_features": missing_features
        }), 400

    input_features = np.array([data[feature] for feature in required_features]).reshape(1, -1)
    
    try:
        prediction = model.predict(input_features)
        probability = model.predict_proba(input_features).tolist()
        
        return jsonify({
            "prediction": int(prediction[0]),
            "probability": probability
        })
    except Exception as e:
        return jsonify({
            "error": "Error during prediction",
            "message": str(e)
        }), 500
        
@app.route('/', methods=['GET'])
def saludo():
    return "Endpoint prueba funcional"

if __name__ == '__main__':
    app.run(debug=True)
