from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained machine learning model
model = joblib.load('insurance_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values from the form
        age = float(request.form['age'])
        bmi = float(request.form['bmi'])
        smoker = int(request.form['smoker'])
        region = request.form['region']

        # Map region to numerical value (you might need to adjust this based on your encoding)
        regions = {'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3}
        region_encoded = regions[region]

        # Prepare input features as a numpy array
        input_data = np.array([[age, bmi, smoker, region_encoded]])

        # Predict insurance price
        predicted_price = model.predict(input_data)

        # Render the prediction result
        return render_template('result.html', predicted_price=predicted_price[0])

if __name__ == '__main__':
    app.run(debug=True)