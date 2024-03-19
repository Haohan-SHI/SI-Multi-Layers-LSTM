import os
import json
import time
import math
import matplotlib.pyplot as plt
import numpy as np
from core.data_processor import DataLoader
from core.model import Model
from keras.utils.vis_utils import plot_model


# 绘图展示结果
def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.grid()
    plt.show()
    #plt.savefig('results_2.png')


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.legend()
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
    plt.grid()
    plt.show()
    #plt.savefig('results_multiple_2.png')

#RNN时间序列

#读取所需参数
configs = json.load(open('640-80-4-layer.json', 'r'))
if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])
#读取数据
data = DataLoader(
    os.path.join('data', configs['data']['visual_filename']),
    configs['data']['visualization_full_split'], # change the train_test_split value to load the full audio file as an input.
    configs['data']['visual_filename_columns']
)


print(data)

#创建RNN模型
model = Model()
mymodel = model.build_model(configs)

model_path = '/home/shh/Desktop/LSTM_Final/saved_models/(640_80_50epo_3layers)02022023_640_80_50epo-e50.h5'
model.load_model(filepath = model_path)
plot_model(mymodel, to_file='model.png',show_shapes=True)


#测试结果
x_test, y_test = data.get_test_data(
    seq_len=configs['data']['sequence_length'],
    normalise=False
    #normalise=configs['data']['normalise']
)


#展示测试效果
predictions_multiseq = model.predict_sequences_multiple(data=x_test, window_size=640, prediction_len=16,debug=False)
#predictions_pointbypoint = model.predict_point_by_point(x_test,debug=False)        

plot_results_multiple(predictions_multiseq, y_test, prediction_len=16)
#plot_results(predictions_pointbypoint, y_test)

print(predictions_multiseq)