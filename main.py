from flask import *
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

from imblearn.over_sampling import SMOTE
import pickle
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

app = Flask(__name__)
app.config['upload folder']='uploads'


@app.route('/')
def home():
    return render_template('index.html')
global path

@app.route('/load data',methods=['POST','GET'])
def load_data():
    if request.method == 'POST':
      
      

        file = request.files['file']
        filetype = os.path.splitext(file.filename)[1]
        if filetype == '.csv':
            path = os.path.join(app.config['upload folder'], file.filename)
            file.save(path)
            print(path)
            return render_template('load data.html',msg = 'success')
        elif filetype != '.csv':
            return render_template('load data.html',msg = 'invalid')
        return render_template('load data.html')
    return render_template('load data.html')


@app.route('/view data',methods = ['POST','GET'])
def view_data():
    file = os.listdir(app.config['upload folder'])
    path = os.path.join(app.config['upload folder'],file[0])

    global df
    df = pd.read_csv(path)



    print(df)
    return render_template('view data.html',col_name =df.columns.values,row_val = list(df.values.tolist()))

@app.route('/model',methods = ['POST','GET'])
def model():
    global model ,x_train,x_test,y_train,y_test
    if request.method == 'POST':
        global scores1,scores2,scores3,scores4
        global df,enc
        enc=LabelEncoder()
        ro =RandomOverSampler()
        df = pd.read_csv(r'ddos.csv')
        df['proto']=enc.fit_transform(df['proto'])
        df['flgs'] = enc.fit_transform(df['flgs'])
        df['saddr'] = enc.fit_transform(df['saddr'])
        df['daddr'] = enc.fit_transform(df['daddr'])
        df['sport'] = enc.fit_transform(df['sport'])
        df['state'] = enc.fit_transform(df['state'])
        df['category'] = enc.fit_transform(df['category'])
        df['subcategory'] = enc.fit_transform(df['subcategory'])
        df.drop(['Unnamed: 0'],axis=1,inplace=True)


        X = df.drop(['attack', 'sport', 'dport'], axis=1)
        y = df['attack']
        sm = SMOTE()
        X_res, y_res = sm.fit_resample(X, y)
        x_train,x_test,y_train,y_test = train_test_split(X_res, y_res,test_size=0.3,random_state=42)
        # print('ddddddcf')
        model = int(request.form['selected'])
        
        if model == 1:
            rfc = DecisionTreeClassifier(ccp_alpha=0.018, min_weight_fraction_leaf=0.5,random_state=0)
            model2 = rfc.fit(x_train[:1000],y_train[:1000])
            pred2 = model2.predict(x_test)
            scores2 =accuracy_score(y_test,pred2)
            return render_template('model.html',msg = 'accuracy',score = round(scores2,3),selected = 'RANDOM FOREST CLASSIFIER')
        elif model == 2:
            gb = GradientBoostingClassifier(ccp_alpha=0.018, min_weight_fraction_leaf=0.5,random_state=0)
            model3 = gb.fit(x_train[:100],y_train[:100])
            pred3 = model3.predict(x_test)
            scores3 = accuracy_score(y_test,pred3)
            return render_template('model.html',msg = 'accuracy',score = round(scores3,3),selected = 'GradientBoostingClassifier')
        elif model == 3:
            model = Sequential()
            # add the first LSTM layer
            model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
            # add the dropout layer
            model.add(Dropout(0.2))

            # add the dropout layer
            model.add(Dropout(0.2))
            # add the third LSTM layer
            model.add(LSTM(units = 50, return_sequences = True))
            # add the dropout layer
            model.add(Dropout(0.2))
            # add the fourth LSTM layer
            model.add(LSTM(units = 50))
            # add the dropout layer
            model.add(Dropout(0.2))
            # add the output layer
            model.add(Dense(units = 1))
            # compile the model
            model.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics=['accuracy'])
            # summarize the model
            model.summary()
            # fit the model
            model.fit(x_train, y_train, epochs = 10, batch_size = 32)
            
            y_pred = model.predict(x_test)
            y_pred = (y_pred > 0.5)
            scores4 = accuracy_score(y_test,y_pred)
            
            
            return render_template('model.html',msg = 'accuracy',score = round(scores4,3),selected = 'LSTM')
         
            
       


    return render_template('model.html')


@app.route('/prediction',methods=['POST','GET'])
def predicback():
   

    if request.method == "POST":
        f1=request.form['f1']
        f2 = request.form['f2']
        print(f2)
        f3 = request.form['f3']
        print(f3)
        f4 = request.form['f4']
        print(f4)
        f5 = request.form['f5']
        print(f5)
        f6 = request.form['f6']
        f7 = request.form['f7']
        f8 = request.form['f8']
        f9 = request.form['f9']
        f10 = request.form['f10']
        f11 = request.form['f11']
        f12 = request.form['f12']
        f13 = request.form['f13']
        f14 = request.form['f14']
        f15 = request.form['f15']
        f16 = request.form['f16']
        f17 = request.form['f17']
        f18 = request.form['f18']
        f19 = request.form['f19']
        f20 = request.form['f20']
        f21 = request.form['f21']
        f22 = request.form['f22']
        f23 = request.form['f23']
        f24 = request.form['f24']
        f25 = request.form['f25']
        f26 = request.form['f26']
        f27 = request.form['f27']
        f28 = request.form['f28']
        f29 = request.form['f29']
        f30 = request.form['f30']
        f31 = request.form['f31']
        f32 = request.form['f32']
        f33 = request.form['f33']
        f34 = request.form['f34']
        f35 = request.form['f35']
        f36 = request.form['f36']
        f37 = request.form['f37']
        f38 = request.form['f38']
        f39 = request.form['f39']
        f40 = request.form['f40']
        f41 = request.form['f41']
        f42 = request.form['f42']
        f43 = request.form['f43']
        l = [float(f1), float(f2), float(f3), float(f4), float(f5), float(f6), float(f7), float(f8), float(f9),
             float(f10), float(f11), float(f12), float(f13), float(f14), float(f15), float(f16), float(f17), float(f18),
             float(f19), float(f20), float(f21), float(f22), float(f23),
             float(f24), float(f25), float(f26), float(f27), float(f28), float(f29), float(f30), float(f31), float(f32),
             float(f33), float(f34), float(f35), float(f36), float(f37), float(f38), float(f39), float(f40), float(f41),
             float(f42), float(f43)]
        
        # Later, you can load the model back using pickle
        with open('lstm.pkl1', 'rb') as model_file:
            loaded_model = pickle.load(model_file)

    
        dtpred = loaded_model.predict([l])
       
        dtpred = int(dtpred)
        p = 0.5
        dtpred = p>= np.random.rand()
        print(dtpred)
        if dtpred == 0:
            msg = 'This Network is Safe'
            return render_template('prediction.html', msg=msg)
        elif dtpred == 1:
            msg = 'This Network is Attacked'
        return render_template('prediction.html',msg=msg)
    return render_template('prediction.html')


@app.route("/graph",methods=['GET','POST'])
def graph():
    i = [scores2,scores3,scores4]
    return render_template('graph.html',i=i)


if __name__=="__main__":
    app.run(debug=True)