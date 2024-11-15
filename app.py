from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# Load models
working_dir = os.path.dirname(os.path.abspath(__file__))
diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open(f'{working_dir}/saved_models/parkinsons_model.sav', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    result = None
    if request.method == 'POST':
        user_input = [
            float(request.form['pregnancies']),
            float(request.form['glucose']),
            float(request.form['blood_pressure']),
            float(request.form['skin_thickness']),
            float(request.form['insulin']),
            float(request.form['bmi']),
            float(request.form['dpf']),
            float(request.form['age'])
        ]
        prediction = diabetes_model.predict([user_input])
        result = 'Diabetic' if prediction[0] == 1 else 'The Person is not Diabetic'
    return render_template('diabetes.html', result=result)

@app.route('/heart', methods=['GET', 'POST'])
def heart():
    result = None
    if request.method == 'POST':
        user_input = [float(request.form[field]) for field in [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]]
        prediction = heart_disease_model.predict([user_input])
        result = 'Heart Disease Detected' if prediction[0] == 1 else 'The person does not have any Heart Diseases'
    return render_template('heart.html', result=result)

@app.route('/parkinsons', methods=['GET', 'POST'])
def parkinsons():
    result = None
    if request.method == 'POST':
        user_input = [float(request.form[field]) for field in [
            'fo', 'fhi', 'flo', 'jitter_percent', 'jitter_abs', 'rap', 'ppq', 'ddp',
            'shimmer', 'shimmer_db', 'apq3', 'apq5', 'apq', 'dda', 'nhr', 'hnr',
            'rpde', 'dfa', 'spread1', 'spread2', 'd2', 'ppe'
        ]]
        prediction = parkinsons_model.predict([user_input])
        result = "Parkinson's Detected" if prediction[0] == 1 else "The person does not have Parkinson's Disease"
    return render_template('parkinsons.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)