from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = float(request.form['age'])
        experience = float(request.form['experience'])

        # Create a DataFrame for input
        data = pd.DataFrame({
            'Age': [age],
            'Years_of_Experience': [experience]
        })

        # Make a salary prediction
        prediction = model.predict(data)

        # Display the prediction
        return render_template('index.html', prediction="Predicted Salary: ${:,.2f}".format(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True)
