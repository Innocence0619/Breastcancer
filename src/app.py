from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from metaflow import Flow
from metaflow import get_metadata, metadata

FLOW_NAME = 'BreastCancerFlow'  # name of the target class that generated the model
# Set the metadata provider as the src folder in the project,
# which should contains /.metaflow
metadata('../src')
# Fetch currently configured metadata provider to check it's local!
print(get_metadata())


def get_latest_successful_run(flow_name: str):
    "Gets the latest successfull run."
    for r in Flow(flow_name).runs():
        if r.successful:
            return r


# get artifacts from latest run, using Metaflow Client API
latest_run = get_latest_successful_run(FLOW_NAME)
latest_pre1 = latest_run.data.standard
latest_pre2 = latest_run.data.modelpca
latest_model = latest_run.data.model

# We need to initialise the Flask object to run the flask app
# By assigning parameters as static folder name,templates folder name
app = Flask(__name__, static_folder='static', template_folder='templates')


@app.route('/', methods=['POST', 'GET'])
def main():
    # on GET we display the page
    if request.method == 'GET':
        return render_template('test.html', project=FLOW_NAME)
    # on POST we make a prediction over the input text supplied by the user
    if request.method == 'POST':
        # debug
        # print(request.form.keys())
        _radius_mean = request.form['_radius_mean']
        _texture_mean = request.form['_texture_mean']
        _perimeter_mean = request.form['_perimeter_mean']
        _area_mean = request.form['_area_mean']
        _smoothness_mean = request.form['_smoothness_mean']
        _compactness_mean = request.form['_compactness_mean']
        _concavity_mean = request.form['_concavity_mean']
        _concavepoints_mean = request.form['_concavepoints_mean']
        _symmetry_mean = request.form['_symmetry_mean']
        _fractal_dimension_mean = request.form['_fractal_dimension_mean']
        _radius_se = request.form['_radius_se']
        _texture_se = request.form['_texture_se']
        _perimeter_se = request.form['_perimeter_se']
        _area_se = request.form['_area_se']
        _smoothness_se = request.form['_smoothness_se']
        _compactness_se = request.form['_compactness_se']
        _concavity_se = request.form['_concavity_se']
        _concavepoints_se = request.form['_concavepoints_se']
        _symmetry_se = request.form['_symmetry_se']
        _fractal_dimension_se = request.form['_fractal_dimension_se']
        _radius_worst = request.form['_radius_worst']
        _texture_worst = request.form['_texture_worst']
        _perimeter_worst = request.form['_perimeter_worst']
        _area_worst = request.form['_area_worst']
        _smoothness_worst = request.form['_smoothness_worst']
        _compactness_worst = request.form['_compactness_worst']
        _concavity_worst = request.form['_concavity_worst']
        _concavepoints_worst = request.form['_concavepoints_worst']
        _symmetry_worst = request.form['_symmetry_worst']
        _fractal_dimension_worst = request.form['_fractal_dimension_worst']
        arr = np.array([float(_radius_mean), float(_texture_mean), float(_perimeter_mean), float(_area_mean), float(_smoothness_mean), float(_compactness_mean), float(_concavity_mean), float(_concavepoints_mean),
                                      float(_symmetry_mean), float(_fractal_dimension_mean), float(_radius_se), float(_texture_se),float(_perimeter_se), float(_area_se),float(_smoothness_se), float(_compactness_se), float(_concavity_se), float(_concavepoints_se), float(_symmetry_se),
                                      float(_fractal_dimension_se), float(_radius_worst), float(_texture_worst), float(_perimeter_worst),float(_area_worst), float(_smoothness_worst), float(_compactness_worst), float(_concavity_worst), float(_concavepoints_worst), float(_symmetry_worst), float(_fractal_dimension_worst)])
        X = pd.DataFrame(arr.reshape(1,30))
        X.columns =['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concavepoints_mean','symmetry_mean','fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se',
                                                    'concavepoints_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concavepoints_worst','symmetry_worst','fractal_dimension_worst']
        X_std = latest_pre1.transform(X)
        X_train = latest_pre2.transform(X_std)
        val = latest_model.predict(X_train)
        if val == 0:
            val1 = "Benign"
        else:
            val1 = "Malignant"
        #  debug
        print('prediction', val)
        # Returning the response to the client
        return "Predicted Y is {}".format(val1)


if __name__ == '__main__':
    # Run the Flask app to run the server
    app.run(host='0.0.0.0',port=8080)