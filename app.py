
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the trained machine learning model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    MODEL_LOADED = True
    print("Model loaded successfully.")
except Exception as exc:
    model = None
    MODEL_LOADED = False
    print(f"Failed to load model: {exc}")


@app.route("/", methods=["GET"])
def index():
    """Display the car price prediction form."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """Return application health information."""
    status_code = 200 if MODEL_LOADED else 503

    return jsonify({
        "status": "healthy" if MODEL_LOADED else "unhealthy",
        "model_loaded": MODEL_LOADED
    }), status_code


@app.route("/prediction", methods=["POST"])
def prediction():
    """Predict a used car's resale price."""
    if not MODEL_LOADED:
        return "The prediction model is unavailable. Check the application logs.", 503

    try:
        # Read submitted form values
        brand = request.form.get("brand", "").strip()
        year = int(request.form.get("year", ""))
        km_driven = float(request.form.get("km_driven", ""))
        owner = request.form.get("owner", "").strip()
        fuel = request.form.get("fuel", "").strip()
        transmission = request.form.get("transmission", "").strip()

        # Validate the submitted values
        if not all([brand, owner, fuel, transmission]):
            return "Please fill in all fields.", 400

        if year < 1900 or year > 2100 or km_driven < 0:
            return "Please enter a valid year and kilometre reading.", 400

        # Match the feature names used during model training
        car_data = pd.DataFrame([{
            "year": year,
            "km_driven": km_driven,
            "owner": owner,
            "fuel": fuel,
            "transmission": transmission,
            "brand": brand
        }])

        predicted_price = max(0.0, float(model.predict(car_data)[0]))

        # Keep the submitted values and show the prediction
        return render_template(
            "index.html",
            prediction=f"₹{predicted_price:,.2f}",
            brand=brand,
            year=year,
            km_driven=km_driven,
            owner=owner,
            fuel=fuel,
            transmission=transmission
        )

    except (ValueError, TypeError):
        return "Please enter valid values for all fields.", 400

    except Exception as exc:
        app.logger.exception("Prediction failed")
        return f"Prediction failed: {exc}", 500


if __name__ == "__main__":
    # Required so Docker can expose the Flask application
    app.run(host="0.0.0.0", port=5000, debug=False)
