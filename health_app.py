from flask import Flask,render_template,request
from tensorflow.keras.models import load_model
import keras
import numpy as np
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
import os 
import cv2
from keras.applications.vgg19 import preprocess_input
from tensorflow.keras.applications import  VGG19
vgg16 = VGG19(include_top=False,weights='imagenet')


new_model =load_model("saved_model/skin_model")

UPLOAD_FOLDER="C:/Users/Agras/Desktop/flask_health/uploads"



posts=[{
    'name':'aditi',
    'age':10,
}]

app = Flask(__name__)
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER



@app.route("/")
def hello_world():
    return render_template('home.html',title='Home')

@app.route("/cold_form")
def cold_form():
    return render_template('cold_form.html',title='Cold_test')


@app.route("/headache_form")
def headache_form():
    return render_template('headache_form.html',title='headache_test')


@app.route("/skin_form")
def skin_form():
    return render_template('skin_form.html',title='skin_test')


@app.route("/cure")
def cureM():
    return render_template('migrainecure.html',title='cure')


@app.route("/pic")
def pic():
    return send_file("HouseCare.png", mimetype='image/gif')


@app.route("/about")
def about():
    return render_template('about.html',title='About')
@app.route("/contact")
def contact():
    return render_template('contact.html',title='Contact')

#@app.route("/skin_form")
#def skin():
 #   return render_template('skin.html',title='Skin_disease')

@app.route("/result_cold",methods=['POST','GET'])
def result_cold():
    int_features=[ x for x in request.form.values()]
    flag=0
    ct=0

    for i in int_features:
        if i=="covid":
            ct+=1
        else:
            flag+=1
        if ct>=flag:
            state="covid"
        else:
            state="cold"

    return render_template('result_form.html',posts=state,title='Result-Cold')



@app.route("/result_headache",methods=['POST','GET'])
def result_headache():
    int_features=[ x for x in request.form.values()]
    state=int_features[0]
    return render_template('result_form.html',posts=state,title='Result-Cold')



@app.route("/predict",methods=['POST','GET'])
def predict():

    def load_img(img_path):
        images=[]
        #img_path='uploads/img1.jpg'
        img=cv2.imread(img_path)
        img=cv2.resize(img,(100,100))
        images.append(img)
        x_test=np.asarray(images)
        test_img=preprocess_input(x_test)
        features_test=vgg16.predict(test_img)
        num_test=x_test.shape[0]
        f_img=features_test.reshape(num_test,4608)
        result=np.argmax(new_model.predict(f_img))
        train_list_mod=['Acne and Rosacea', 
        'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions', 
        'Atopic Dermatitis', 
        'Eczema', 
        'Nail Fungus and other Nail Disease', 
        'Psoriasis pictures Lichen Planus and related diseases']
        f_result=train_list_mod[result]

        return f_result

    img_path=request.form.get('path')
    j=load_img(img_path)
    return render_template('result_skin.html',posts=j)


if __name__=='__main__':
    app.run(debug=True)