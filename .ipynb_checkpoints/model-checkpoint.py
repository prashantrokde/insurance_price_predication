import joblib

class InsuranceModel:
    def __init__(self, model_path):
        # Load the trained machine learning model
        self.model = joblib.load(model_path)

    def predict_insurance_price(self, age, bmi, smoker, region):
        # Map region to numerical value (you might need to adjust this based on your encoding)
        regions = {'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3}
        region_encoded = regions[region]

        # Prepare input features as a numpy array
        input_data = [[age, bmi, smoker, region_encoded]]

        # Predict insurance price
        predicted_price = self.model.predict(input_data)

        return predicted_price[0]
