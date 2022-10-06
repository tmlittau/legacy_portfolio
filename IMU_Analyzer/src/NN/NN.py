import os
import json
import time

import tensorflow as tf
import numpy as np
import math

from data_loader import DataLoader
from model import Model

import matplotlib.pyplot as plt


class LSTM():

    def __init__(self, filename):
        self.configs = json.load(open('config.json', 'r'))
        self.configs['data']['filename'] = filename

        if not os.path.exists(self.configs['model']['save_dir']):
            os.makedirs(self.configs['model']['save_dir'])

        self.data = DataLoader(
            os.path.join('data', self.configs['data']['filename']),
            self.configs['data']['train_test_split'],
            self.configs['data']['columns'],
            self.configs['data']['output_dimension']
        )

        self.model = Model()

    def update_filename(self, filename):
        self.configs['data']['filename'] = filename

    def training(self, Window):
        self.configs['training']['batch_size'] = Window.batch_box.value()
        self.configs['training']['epochs']     = Window.epoch_box.value()
        
        print('Initialize Data for Training')
        self.data = DataLoader(
            os.path.join('data', self.configs['data']['filename']),
            self.configs['data']['train_test_split'],
            self.configs['data']['columns'],
            self.configs['data']['output_dimension']
        )
        
        self.model.build_model(self.configs)
        Window.write_to_terminal('[Model] Model Compiled')
        #x, y = self.data.get_train_data(
        #    seq_len = self.configs['data']['sequence_length'],
        #    normalise = self.configs['data']['normalise']
        #)

        steps_per_epoch = math.ceil((self.data.len_train - self.configs['data']['sequence_length']) / self.configs['training']['batch_size'])

        Window.init_progbar(self.data.len_train)
        
        model_file = self.model.train_generator(
            data_gen = self.data.generate_train_batch(
                seq_len = self.configs['data']['sequence_length'],
                batch_size = self.configs['training']['batch_size'],
                normalise = self.configs['data']['normalise']
            ),
            epochs = self.configs['training']['epochs'],
            batch_size = self.configs['training']['batch_size'],
            steps_per_epoch = steps_per_epoch,
            save_dir = self.configs['model']['save_dir'],
            window = Window
        )

        return model_file

    def testing(self, Window):
        x_test, y_test = self.data.get_test_data(
            seq_len=self.configs['data']['sequence_length'],
            normalise = self.configs['data']['normalise']
        )

        Window.write_to_terminal('[Model] Predicting Point-by-Point...')
        prediction = self.model.predict_point_by_point(x_test)
        
        
        fig = self.plot_results(prediction, y_test, x_test)

        Window.write_to_terminal('[Model] Prediction Completed.')
        return fig

    def testing_whole(self, Window):
        Window.write_to_terminal('[Model] Loading Data...')
        
        self.data = DataLoader(
            os.path.join('data', self.configs['data']['filename']),
            self.configs['data']['train_test_split'],
            self.configs['data']['columns'],
            self.configs['data']['output_dimension']
        )
        
        x, y = self.data.get_all_data(
            seq_len=self.configs['data']['sequence_length'],
            normalise = self.configs['data']['normalise']
        )
        error_lin = []
        error_ang = []
        rmse_lin = 0.0
        rmse_ang = 0.0
        
        Window.write_to_terminal('[Model] Predicting Point-by-Point...')
        prediction = self.model.predict_point_by_point(x)
        for tt in range(len(x)):
            error_lin.append(abs(prediction[tt,0]-y[tt,0]))
            error_ang.append(abs(prediction[tt,1]-y[tt,1]))
            rmse_lin = rmse_lin + error_lin[tt]**2
            rmse_ang = rmse_ang + error_ang[tt]**2

        rmse_lin = math.sqrt(rmse_lin / (len(x)/0.02))
        rmse_ang = math.sqrt(rmse_ang / (len(x)/0.02))

        Window.write_to_terminal('[Prediction] RMSE linear = {}'.format(rmse_lin))
        Window.write_to_terminal('[Prediction] RMSE angular = {}'.format(rmse_ang))
        Window.write_to_terminal('[Model] Prediction Completed.')

        #plt.plot(error_lin, label = 'Acc-Error')
        #plt.plot(error_ang, label = 'Vel-Error')
        #plt.legend()
        #plt.show()

        
        fig = self.plot_results_load(prediction, y, x)

        return fig

    def plot_results(self, prediction, true_data, xx):
        
        f0 = plt.figure(1)
        '''
        plt.subplot(221)
        plt.plot(xx[:,0,1], 'C2', label= 'True Data')
        plt.title('Linear Command')
        plt.xlabel('time in [ms]')
        plt.ylabel('Velocity in [s]')
        '''
        plt.subplot(211)
        plt.plot(true_data[:,0], 'C2', label= 'True Data')
        plt.plot(prediction[:,0], 'C1', label='Prediction')
        plt.title('Linear Acceleration X')
        plt.xlabel('time in [ms]')
        plt.ylabel('Velocity in [s]')
        plt.legend()
        '''
        plt.subplot(223)
        plt.plot(xx[:,0,0], 'C2', label= 'True Data')
        plt.title('Angular Command')
        plt.xlabel('time in [ms]')
        plt.ylabel('Velocity in [s]')
        '''
        plt.subplot(212)
        plt.plot(true_data[:,1], 'C2', label= 'True Data')
        plt.plot(prediction[:,1], 'C1', label='Prediction')
        plt.title('Angular Velocity Z')
        plt.xlabel('time in [ms]')
        plt.ylabel('Velocity in [s]')
        plt.legend()
     
        f0.tight_layout()

        plt.show()

    def plot_results_load(self, prediction, true_data, xx):
        
        f0 = plt.figure(1)
 
        plt.subplot(211)
        plt.plot(true_data[:,0], 'C2', label= 'True Data')
        plt.plot(prediction[:,0], 'C1', label='Prediction')
        plt.title('Linear Acceleration X')
        plt.xlabel('time in [ms]')
        plt.ylabel('Acceleration in [m/s$^2$]')
        plt.legend()

        plt.subplot(212)
        plt.plot(true_data[:,1], 'C2', label= 'True Data')
        plt.plot(prediction[:,1], 'C1', label='Prediction')
        plt.title('Angular Velocity Z')
        plt.xlabel('time in [ms]')
        plt.ylabel('Velocity in [m/s]')
        plt.legend()

        '''
        plt.subplot(223)
        plt.plot(xx[:,0,2], 'C2', label= 'True Data')
        plt.title('Linear Command X')
        plt.xlabel('time in [ms]')
        plt.ylabel('Velocity in [s]')

        plt.subplot(214)
        plt.plot(xx[:,0,0], 'C2', label= 'True Data')
        plt.title('Angular Command Z')
        plt.xlabel('time in [ms]')
        plt.ylabel('Velocity in [s]')     
        '''
        f0.tight_layout()

        plt.show()

    def __del__(self):
        tf.keras.backend.clear_session()
