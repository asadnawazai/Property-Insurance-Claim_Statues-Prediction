from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open(r"C:\Users\Dell User\Desktop\Client-Dataset\2nd-Gradient-Boosting.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def form():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    try:
        claim_amount = float(request.form['Claim_Amount'])
        building_age = int(request.form['Building_Age'])
        repair_estimate = float(request.form['Repair_Estimate'])
        policy_coverage_amount = float(request.form['Policy_Coverage_Amount'])
        amount_paid = float(request.form['Amount_Paid'])
        property_type = request.form['Property_Type']
        property_location = request.form['Property_Location']
        incident_type = request.form['Incident_Type']
        severity = request.form['Severity']
        building_material = request.form['Building_Material']
        fraudulent = request.form['Fraudulent']
    except ValueError:
        return "Invalid input data. Please check your inputs and try again."

    # Prepare input data as a DataFrame
    input_data = pd.DataFrame([{
        'Claim_Amount': claim_amount,
        'Building_Age': building_age,
        'Repair_Estimate': repair_estimate,
        'Policy_Coverage_Amount': policy_coverage_amount,
        'Amount_Paid': amount_paid,
        'Property_Type': property_type,
        'Property_Location': property_location,
        'Incident_Type': incident_type,
        'Severity': severity,
        'Building_Material': building_material,
        'Fraudulent': fraudulent
    }])

    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Map prediction to result category
    if prediction == 1:
        result = "Approved"
    elif prediction == 0:
        result = "Rejected"
    else:
        result = "Pending"

    return render_template('form.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
