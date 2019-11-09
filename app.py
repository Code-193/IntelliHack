from flask import Flask,render_template,request,url_for
import numpy as np
from keras.models import load_model
 
model = load_model('paddy.h5')
l=[0.35361, -1.35483, 0.803285, -0.52981,-1.08189, 1.162138]

l=np.asarray(l)
l=np.reshape(l,(1,6))
print(l.shape)
y_pred=model.predict(l)
print(y_pred)


app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/result',methods=['GET','POST'])
def result():
    if(request.method=="POST"):
        ph=request.form['ph']
        slevel=request.form['slevel']
        temp=request.form['temp']
        season=request.form['season']
        humidity=request.form['humidity']
        variety=request.form['variety']
        l=[(float(slevel)-1260.999)/720.3919,(float(temp)-27.55163)/7.23273,(float(ph)-6.01819)/1.737971,int(season),(float(humidity)-76.9539)/10.0742,(float(variety)-3.4)/1.95969]
        #l=[int(slevel),temp,ph,season,humidity,variety]
        print(l)
        l=np.asarray(l)
        l=np.reshape(l,(1,6))
        print(l.shape)
        y_pred=model.predict(l)
        print(y_pred)
    return render_template('index.html',output=y_pred,ph=ph,slevel=slevel,temp=temp,season=season,humidity=humidity,variety=variety)

app.run()



