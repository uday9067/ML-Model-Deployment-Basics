from flask import Flask, request, jsonify, render_template
import pickle
import json
import numpy as np

app = Flask(__name__, static_folder='static', template_folder='templets')

# Load model & columns
model = pickle.load(open("banglore_home_prices_model.pickle", "rb"))
with open("columns.json", "r") as f:
    data_columns = json.load(f)['data_columns']

@app.route('/')
def home():
    return render_template('index.html')  # UI form (optional)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        # Get data from form or JSON
        total_sqft = float(request.form['total_sqft'])
        bath = int(request.form['bath'])
        bhk = int(request.form['bhk'])
        location = request.form['location']

        # Prepare input array
        x = np.zeros(len(data_columns))
        x[0] = total_sqft
        x[1] = bath
        x[2] = bhk
        if location.lower() in data_columns:
            loc_index = data_columns.index(location.lower())
            x[loc_index] = 1

        # Predict
        prediction = model.predict([x])[0]
        # return jsonify({'estimated_price_lakh': round(prediction, 2)})
        return render_template('predict.html',predicted_price=round(prediction, 2))

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
