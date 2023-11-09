from flask import Flask,request,render_template
import numpy as np
import pandas as pd

import pdb

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    results=None
    if request.method=='GET':
        return render_template('home.html', results=0)
    else:
        data = CustomData(
            surgery=request.form.get('surgery'),
            age=request.form.get('age'),
            hospital_number=request.form.get('hospital_number'),
            rectal_temp=float(request.form.get('rectal_temp')),
            pulse=float(request.form.get('pulse')),
            respiratory_rate=float(request.form.get('respiratory_rate')),
            temp_of_extremities=request.form.get('temp_of_extremities'),
            peripheral_pulse=request.form.get('peripheral_pulse'),
            mucous_membrane=request.form.get('mucous_membrane'),
            capillary_refill_time=request.form.get('capillary_refill_time'),
            pain=request.form.get('pain'),
            peristalsis=request.form.get('peristalsis'),
            abdominal_distention=request.form.get('abdominal_distention'),
            nasogastric_tube=request.form.get('nasogastric_tube'),
            nasogastric_reflux=request.form.get('nasogastric_reflux'),
            nasogastric_reflux_ph=float(request.form.get('nasogastric_reflux_ph')),
            rectal_exam_feces=request.form.get('rectal_exam_feces'),
            abdomen=request.form.get('abdomen'),
            packed_cell_volume=request.form.get('packed_cell_volume'),
            total_protein=request.form.get('total_protein'),
            abdomo_appearance=request.form.get('abdomo_appearance'),
            abdomo_protein=request.form.get('abdomo_protein'),
            surgical_lesion=request.form.get('surgical_lesion'),
            lesion_1=request.form.get('lesion_1'),
            lesion_2=request.form.get('lesion_2'),
            cp_data=request.form.get('cp_data')
        )

        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)

        return render_template('home.html',results=results)
    

if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True) 