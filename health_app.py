from flask import Flask,render_template,request
import numpy as np
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
import os 
import pickle
import sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


model=pickle.load(open('saved_model/model.pkl','rb'))


#lung cancer rate prediction over a given value of aqi.


app = Flask(__name__)



@app.route("/")
def hello_world():
    return render_template('home.html',title='Home')

@app.route("/lung_cancer")
def cold_form():
    return render_template('lung_cancer.html',title='Cold_test')


@app.route("/cure")
def cureM():
    return render_template('migrainecure.html',title='cure')

@app.route("/about")
def about():
    return render_template('about.html',title='About')
@app.route("/contact")
def contact():
    return render_template('contact.html',title='Contact')


@app.route("/results",methods=['POST','GET'])
def results():
    int_features=[[ x for x in request.form.values()]]
    state=model.predict(int_features)

    return render_template('result_form.html',posts=state,title='Results')


if __name__=='__main__':
    app.run(debug=True)