"""
The CINC data is provided by https://physionet.org/challenge/2017/ 
"""
from __future__ import division, print_function
import numpy as np
from config import get_config
from utils import *
import os 
import json

def cincData(config):
  try:
    if config.cinc_download:
      cmd = "curl -O https://archive.physionet.org/challenge/2017/training2017.zip"
      os.system(cmd)
      os.system("unzip training2017.zip")
    num = config.num
    import csv
    testlabel = []

    with open('training2017/REFERENCE.csv') as csv_file:
      csv_reader = csv.reader(csv_file, delimiter=',')
      line_count = 0
      for row in csv_reader:
        testlabel.append([row[0],row[1]])
        #print(row[0], row[1])
        line_count += 1
      print(f'Processed {line_count} lines.')
    if num == None:
      high = len(testlabel)-1
      num = np.random.randint(1,high)
    filename , label = testlabel[num-1]
    filename = 'training2017/'+ filename + '.mat'
    from scipy.io import loadmat
    data = loadmat(filename)
    print("The record of "+ filename)
    if not config.upload:
        data = data['val']
        _, size = data.shape
        data = data.reshape(size,)
    else:
        data = np.array(data)
    return data, label
  except:
    print("cincData")


def predictTestSampleCSV(data, peaks):
  
  try:
   
    # classesM = ['N','Ventricular','Paced','A','F','Noise']
    classesM = ['N','V','/','A','L','R','S','F','~','f','j','E','a']
    predicted, result, counter  = predictByPartTestSampleCSV(data, peaks)
    sumPredict = sum(predicted[x][1] for x in range(len(predicted)))
    avgPredict = sumPredict/len(predicted)
    print("The average of the predict is:", avgPredict)
    print("The most predicted label is {} with {:3.1f}% certainty".format(classesM[avgPredict.argmax()], 100*max(avgPredict[0])))
    sec_idx = avgPredict.argsort()[0][-2]
    print("The second predicted label is {} with {:3.1f}% certainty".format(classesM[sec_idx], 100*avgPredict[0][sec_idx]))
     # print("The original label of the record is " + label)
    print("classesM")
    print(classesM[avgPredict.argmax()], 100*max(avgPredict[0]))
    displayResults = {'avgPredict': avgPredict, 'mostPredictedLabel': classesM[avgPredict.argmax()], 
    'mostPredictedCertainity': 100*max(avgPredict[0]), 'secondMostPredictedLabel': classesM[sec_idx], 'secondMostPredictedCertainitiy': 100*avgPredict[0][sec_idx]}
    return predicted, result, counter, displayResults
  except Exception as ex:
    print("predict")


def predictByPartTestSampleCSV(data, peaks):
  try:
      # print("PREDICT BY PART")
      # print(peaks)


      # classesM = ['N','Ventricular','Paced','A','F','Noise']#,'L','R','f','j','E','a','J','Q','e','S']
      classesM = ['N','V','/','A','L','R','S','F','~','f','j','E','a']
      predicted = list()
      result = ""
      counter = [0]* len(classesM)
      from keras.models import load_model
      basepath = os.path.dirname(__file__)
      file_path = os.path.join(basepath, 'models', "MLII-latest.hdf5")
      model = load_model(file_path)
      config = get_config() 
      for i, peak in enumerate(peaks[3:-1]):
        total_n =len(peaks)
        start, end =  peak-config.input_size//2 , peak+config.input_size//2
        prob = model.predict(data[:, start:end])
        prob = prob[:,0]
        ann = np.argmax(prob)
        counter[ann]+=1
        
        if classesM[ann] != "N":
          print("The {}/{}-record classified as {} with {:3.1f}% certainty".format(i,total_n,classesM[ann],100*prob[0,ann]))
        result += "("+ classesM[ann] +":" + str(round(100*prob[0,ann],1)) + "%)"
        predicted.append([classesM[ann],prob])
        # jsonResult[classesM[ann]].append([classesM[ann] : str(round(100*prob[0,ann],1))])
        # print("JSON")
        # print(jsonResult[classesM[ann]])
        # print("classesM[ann]")
        # print(classesM[ann])

        if classesM[ann] != 'N' and prob[0,ann] > 0.95:
          import matplotlib.pyplot as plt
          plt.plot(data[:, start:end][0,:,0],)
          mkdir_recursive('results')
          plt.savefig('results/hazard-'+classesM[ann]+'.png', format="png", dpi = 300)
          plt.close()
      # result += "{}-N, {}-Venticular, {}-Paced, {}-A, {}-F, {}-Noise".format(counter[0], counter[1], counter[2], counter[3], counter[4], counter[5])
      # result += "{}-N, {}-V, {}-/, {}-A,{}-L,{}-R,{}-S, {}-F, {}-~, {}-f,{}-j,{}-E,{}-a".format(counter[0], counter[1], counter[2], counter[3], counter[4], counter[5], counter[6],counter[7],counter[8],counter[9],counter[10],counter[11],counter[12])
      
      # count = "{}-N, {}-V, {}-/, {}-A,{}-L,{}-R,{}-S, {}-F, {}-~, {}-f,{}-j,{}-E,{}-a".format(counter[0], counter[1], counter[2], counter[3], counter[4], counter[5], counter[6],counter[7],counter[8],counter[9],counter[10],counter[11],counter[12])

      print("PREDICT BY PART - RESULT")
      print(result)
      return predicted, result, counter
  except Exception as e:
    print(e)







    # Older code

def predict(data, label, peaks, config):
  try:
    # classesM = ['N','Ventricular','Paced','A','F','Noise']
    classesM = ['N','V','/','A','L','R','S','F','~','f','j','E','a']
    predicted, result  = predictByPart(data, peaks)
    sumPredict = sum(predicted[x][1] for x in range(len(predicted)))
    avgPredict = sumPredict/len(predicted)
    print("The average of the predict is:", avgPredict)
    print("The most predicted label is {} with {:3.1f}% certainty".format(classesM[avgPredict.argmax()], 100*max(avgPredict[0])))
    sec_idx = avgPredict.argsort()[0][-2]
    print("The second predicted label is {} with {:3.1f}% certainty".format(classesM[sec_idx], 100*avgPredict[0][sec_idx]))
    print("The original label of the record is " + label)
    if config.upload:
      return predicted, classesM[avgPredict.argmax()], 100*max(avgPredict[0])
  except:
    print("predict")

def predictByPart(data, peaks):
  try:
      print("PREDICT BY PART")
      print(peaks)

      # classesM = ['N','Ventricular','Paced','A','F','Noise']#,'L','R','f','j','E','a','J','Q','e','S']
      classesM = ['N','V','/','A','L','R','S','F','~','f','j','E','a']

      predicted = list()
      result = ""
      counter = [0]* len(classesM)
      from keras.models import load_model
      model = load_model('models/MLII-latest.hdf5')
      config = get_config() 
      for i, peak in enumerate(peaks[3:-1]):
        total_n =len(peaks)
        start, end =  peak-config.input_size//2 , peak+config.input_size//2
        prob = model.predict(data[:, start:end])
        prob = prob[:,0]
        ann = np.argmax(prob)
        counter[ann]+=1
        if classesM[ann] != "N":
          print("The {}/{}-record classified as {} with {:3.1f}% certainty".format(i,total_n,classesM[ann],100*prob[0,ann]))
        result += "("+ classesM[ann] +":" + str(round(100*prob[0,ann],1)) + "%)"
        predicted.append([classesM[ann],prob])
        if classesM[ann] != 'N' and prob[0,ann] > 0.95:
          import matplotlib.pyplot as plt
          plt.plot(data[:, start:end][0,:,0],)
          mkdir_recursive('results')
          plt.savefig('results/hazard-'+classesM[ann]+'.png', format="png", dpi = 300)
          plt.close()
      # result += "{}-N, {}-Venticular, {}-Paced, {}-A, {}-F, {}-Noise".format(counter[0], counter[1], counter[2], counter[3], counter[4], counter[5])
      result += "{}-N, {}-V, {}-/, {}-A,{}-L,{}-R,{}-S, {}-F, {}-~, {}-f,{}-j,{}-E,{}-a".format(counter[0], counter[1], counter[2], counter[3], counter[4], counter[5], counter[6],counter[7],counter[8],counter[9],counter[10],counter[11],counter[12])
      print("PREDICT BY PART - RESULT")
      print(result)
      return predicted, result
  except Exception as e:
    print(e)

def main(config):
  try:
    # classesM= ['N','Ventricular','Paced','A','F', 'Noise']#,'L','R','f','j','E','a','J','Q','e','S']
    classesM = ['N','V','/','A','L','R','S','F','~','f','j','E','a']

    if config.upload:
      data = uploadedData(file)
      print("Uploaded file")
    else:
      data, label = cincData(config)
      print("cincData")
    data, peaks = preprocess(data, config)
    # return predict(data, label, peaks, config)
    return predictTestSampleCSV(data, peaks)

  except:
    print("main")

if __name__=='__main__':
  config = get_config()
  main(config)
