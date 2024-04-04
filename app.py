from flask import Flask, render_template, url_for, request
import pandas as pd 
import pickle
import xgboost as xgb

app = Flask(__name__, template_folder="template")

# Load the model from disk
try:
    loaded_model = pickle.load(open('C:\\Users\\AKSHAY\\Documents\\PBL AIR QUALITY\\aqi_XGBreg_model.pkl', 'rb'))
except Exception as e:
    print("Error loading the model:", e)
    loaded_model = None

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if loaded_model:
        try:
            df = pd.read_csv('real_2018.csv')
            my_prediction = loaded_model.predict(df.iloc[:,:-1].values)
            my_prediction = my_prediction.tolist()
            return render_template('result.html', prediction=my_prediction)
        except Exception as e:
            print("Error predicting:", e)
            return "Error predicting"
    else:
        return "Model not loaded"

if __name__ == '__main__':
    app.run(debug=True)
