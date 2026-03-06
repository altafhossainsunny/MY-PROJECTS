from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import(OneHotEncoder,
                                  OrdinalEncoder,
                                  LabelEncoder,
                                  StandardScaler)
application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predictdata", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "GET":
        return render_template('home.html')
    else:
        from src.pipeline.predict_pipeline import PredictPipeline, CustomData

        data = CustomData(
            gender                      = request.form.get('gender'),
            race_ethnicity              = request.form.get('race_ethnicity'),
            parental_level_of_education = request.form.get('parental_level_of_education'),
            lunch                       = request.form.get('lunch'),
            test_preparation_course     = request.form.get('test_preparation_course')
        )

        df = data.get_data_as_dataframe()
        pipeline = PredictPipeline()
        prediction = pipeline.predict(df)

        return render_template('home.html', prediction=prediction)


@app.route("/maintenance", methods=['GET', 'POST'])
def maintenance():
    if request.method == "GET":
        return render_template('maintenance.html')
    else:
        from src.pipeline.predict_pipeline import MaintenancePredictPipeline, MaintenanceData

        data = MaintenanceData(
            footfall    = request.form.get('footfall'),
            tempMode    = request.form.get('tempMode'),
            AQ          = request.form.get('AQ'),
            USS         = request.form.get('USS'),
            CS          = request.form.get('CS'),
            VOC         = request.form.get('VOC'),
            RP          = request.form.get('RP'),
            IP          = request.form.get('IP'),
            Temperature = request.form.get('Temperature')
        )

        df = data.get_data_as_dataframe()
        pipeline = MaintenancePredictPipeline()
        prediction, failure_pct = pipeline.predict(df)

        return render_template('maintenance.html', prediction=prediction, failure_pct=failure_pct)
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False)