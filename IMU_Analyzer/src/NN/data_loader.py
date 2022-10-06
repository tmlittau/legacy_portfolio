import math
import numpy as np
import pandas

from scipy.signal import butter, lfilter
import rosbag

class DataLoader():

    def __init__(self, filename, split, cols, output_dim):
        dataframe = rosbag.Bag(filename)
        i_split   = int(dataframe.get_message_count(topic_filters=['imu_data']) * split)
        self.fs = 50
        self.outindx = []
        for i in range(output_dim):
            self.outindx.append(i)
            
        tmp     = []
        tmp_cmd = []
        tmp_imu = []
        
        for topic, msg, t in dataframe.read_messages(topics='imu_data'):
            
            tmp_imu.append([msg.linear_acceleration.x,
                            msg.angular_velocity.z
            ])

        for topic, msg, t in dataframe.read_messages(topics='cmd_data'):
            tmp_cmd.append([msg.angular.z, msg.linear.x])

        for ii in range(len(tmp_imu)):
            tmp.append(tmp_imu[ii] + tmp_cmd[ii])


        train_tmp        = tmp[:i_split]
        test_tmp         = tmp[i_split:]
        self.data_train        = np.array(train_tmp)
        self.data_test         = np.array(test_tmp)
        self.len_train         = len(self.data_train)
        self.len_test          = len(self.data_test)
        self.len_train_windows = None
        del tmp, tmp_cmd, tmp_imu, train_tmp, test_tmp


    def get_test_data(self, seq_len, normalise):
        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i+seq_len])

        data_windows = np.array(data_windows).astype(float)
        data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows
        
        x = data_windows[:, :-1]
        y = data_windows[:, -1, self.outindx]
        return x, y

    def get_train_data(self, seq_len, normalise):
        data_x = []
        data_y = []
        for i in range(self.len_train -seq_len):
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)

        return np.array(data_x), np.array(data_y)

    def get_all_data(self, seq_len, normalise):
        data_windows = []
        data = np.concatenate((self.data_train, self.data_test))
        for i in range(self.len_train + self.len_test - seq_len):
            data_windows.append(data[i:i+seq_len])

        data_windows = np.array(data_windows).astype(float)
        data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows

        x = data_windows[:, :-1]
        y = data_windows[:, -1, self.outindx]
        return x, y

    def generate_train_batch(self, seq_len, batch_size, normalise):

        i = 0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):

                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len, normalise):
        window = self.data_train[i:i+seq_len]
        window = self.normalise_windows(window, single_window=True)[0] if normalise else window

        x = window[:-1]
        y = window[-1, self.outindx]
        return x,y

    def normalise_windows(self, window_data, single_window=False):
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                if float(window[0,col_i])!= 0:
                    normalised_col = [((float(p) / float(window[0,col_i])) -1) for p in window[:, col_i]]
                else:
                    normalised_col = [((float(p) / float(1)) -1) for p in window[:, col_i]]
                normalised_window.append(normalised_col)
            normalised_window = np.array(normalised_window).T

            normalised_data.append(normalised_window)
        return np.array(normalised_data)

