from flask import Flask,render_template,request
import numpy as np
from keras.models import load_model
 
model = load_model('crop.h5')
l=[0.353610247429042,-1.35482962989927,0.803285062309744,-0.529806359800491,-1.08189244923467,1.16213831847319,-0.764480017252319]
l=np.asarray(l)
l=np.reshape(l,(1,7))

y_pred = model.predict(l)
y_pred = np.argmax(y_pred, axis=1)
print(y_pred)
output=["Sugarcane","Soybeans","Corn"]
print(output[y_pred[0]])

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('classification.html') 

@app.route('/result',methods=['GET','POST'])
def result():
    if(request.method=="POST"):
        ph=request.form['ph']
        n=request.form['n']
        p=request.form['p']
        k=request.form['k']
        depth=request.form['depth']
        rainfall=request.form['rainfall']
        temp=request.form['temp']
        l=[(float(ph)-7.49046)/0.8753706,(float(n)-100.3397)/46.012944,(float(p)-69.714)/28.98846,(float(k)-59.3475)/34.63058,(float(depth)-37.34411)/12.96258,(float(temp)-27.51593)/7.214348,(float(rainfall)-849.0224)/200.7278]
        l=np.asarray(l)
        l=np.reshape(l,(1,7))
        
        y_pred = model.predict(l)
        y_pred = np.argmax(y_pred, axis=1)
        print(y_pred)
        output=["Sugarcane","Soybeans","Corn"]
        print(output[y_pred[0]])
    return render_template('classification.html',output=output[y_pred[0]],ph=ph,n=n,p=p,k=k,depth=depth,rainfall=rainfall,temp=temp)

app.run()



