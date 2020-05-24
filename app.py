# coding=utf-8
from __future__ import division, print_function
import os
import numpy as np

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from predict import *
from utils import *
from config import get_config
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024
global classesM
# classesM = ['N','V','L','R','Paced','A','F']#,'f','j','E','a','J','Q','e','S']
classesM = ['N', 'V', '/', 'A', 'L', 'R', 'S', 'F', '~', 'f', 'j', 'E', 'a']


def model_predict(img_path):
    data = uploadedData(img_path, csvbool=True)
    sr = data[0]

    data = data[1:]
    size = len(data)
    if size > 9001:
        size = 9001
        data = data[:size]
    div = size // 1000
    data, peaks = preprocess(data, config)
    # uncomment to run traning 2017. comment below
    # return predictByPart(data, peaks)

    # uncomment to run sample csv file. comment above
    return predictTestSampleCSV(data, peaks)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

# @app.route('/help/', methods=['GET'])
# def index():
#     # Main page
#     return render_template('help.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():

    try:
        if request.method == 'POST':
            print("INSIDE APP PY POST")
            # Get the file from post request

            f = request.files['file']
            if not f:
                return "No file!"
            basepath = os.path.dirname(__file__)
            mkdir_recursive('uploads')

            file_path = os.path.join(
                basepath, 'uploads', secure_filename(f.filename))
            try:
                print(
                    "You already analyzed the data file. We delete it and re-analyze it!")
                os.remove(file_path)
            except:
                print("The file is new!")
            f.save(file_path)
            # predicted, result = model_predict(file_path)

            # uncomment to try for sample csv file. Comment "predicted, result = main(config)"  that is  the line below
            # add count variable
            predicted, result, counter, displayResults = model_predict(
                file_path)

            # uncomment to find for random file in training 2017 folder. Comment everything above this line upto request.method line
            # predicted, result, counter, displayResults  = main(config)
            length = len(predicted)
            displayTable = []
            rowCount = 1
            for record in predicted:
                resultRow = {'no': rowCount, 'N': '', 'V': '', '/': '', 'A': '', 'L': '',
                             'R': '', 'S': '', 'F': '', '~': '', 'f': '', 'j': '', 'E': '', 'a': ''}
                ann = np.argmax(record[1])
                resultRow[record[0]] = round(100*record[1][0, ann], 1)
                displayTable.append(resultRow)
                rowCount += 1
            print('-------------')
            avg = "The average of the predict is: {}".format(
                displayResults['avgPredict'])
            mostPredicted = "The most predicted label is {} with {:3.1f}% certainty".format(
                displayResults['mostPredictedLabel'], displayResults['mostPredictedCertainity'])
            secondMostPredicted = "The second predicted label is {} with {:3.1f}% certainty".format(
                displayResults['secondMostPredictedLabel'], displayResults['secondMostPredictedCertainitiy'])
            # result = str(length) +" parts of the divided data were estimated as the followings with paired probabilities. \n"+result+"\n"+count
            # print("INSIDE APP PY POST - RESULT")
            # print(result)

            # return render_template('prediction.html', displayTable=displayTable, counter=counter, mostPredicted=mostPredicted, secondMostPredicted=secondMostPredicted)

            return render_template('prediction.html', displayTable=displayTable, counter=counter, avg=avg, mostPredicted=mostPredicted, secondMostPredicted=secondMostPredicted)
        return None
    except Exception as e:
        print(e)


if __name__ == '__main__':
    config = get_config()
    app.run(port=5000, debug=True)

    # Serve the app with gevent
    # http_server = WSGIServer(('', 5000), app)
    # http_server.serve_forever()
    print('Check http://127.0.0.1:5000/ || localhost:5000')


