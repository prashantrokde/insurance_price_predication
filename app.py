from flask import Flask, render_template, request
from model import InsuranceModel

app = Flask(__name__)

# Initialize the InsuranceModel object
model = InsuranceModel('insurancemodelf.pkl')

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

        # Predict insurance price using the model
        predicted_price = model.predict_insurance_price(age, bmi, smoker, region)

        # Render the prediction result
        return render_template('result.html', predicted_price=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)
