from flask import Flask, request, render_template, url_for
import jsonify
import requests
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
#import flasgger
#from flasgger import Swagger
import pickle

app = Flask(__name__)
#Swagger(app)

model = pickle.load(open('polynomial_linearreg2.pkl', 'rb'))

@app.route('/', methods=["GET"])
def Home():
    return render_template('index.html')

standard_to = StandardScaler()
@app.route('/predict', methods=["POST"])
def predict():
    if request.method == 'POST':
        Gender_1 = request.form['Gender_1']
        if (Gender_1=='Male'):
            Gender_1=1
        else:
            Gender_1=0
        Company_Type_1 = request.form['Company_Type_1']
        if (Company_Type_1=='Product'):
            Company_Type_1=1
        else:
            Company_Type_1=0
        WFH_Setup_Available_1 = request.form['WFH_Setup_Available_1']
        if (WFH_Setup_Available_1=='Yes'):
            WFH_Setup_Available_1=1
        else:
            WFH_Setup_Available_1=0
        Designation = float(request.form['Designation'])
        Resource_Allocation = float(request.form['Resource Allocation'])
        Mental_Fatigue_Score = float(request.form['Mental Fatigue Score'])
        inputs = [[Gender_1, Company_Type_1, WFH_Setup_Available_1, Designation, Resource_Allocation, Mental_Fatigue_Score]]
        prediction = model.predict(inputs)
        output=round(prediction[0],2)
        result=pd.cut(output, bins = [0., 0.2, 0.4, 0.6, 0.8, 1.,],labels=['HONEYMOON PHASE', 'ONSET OF STRESS', 'CHRONIC STRESS', 'BURNOUT', 'HABITUAL BURNOUT'])
        if inputs<0:
            return render_template('index.html',prediction_texts="Sorry, please check the data entered")
        else:
            return render_template('index.html',prediction_text="Employee is experiencing {}".format(result))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)

        