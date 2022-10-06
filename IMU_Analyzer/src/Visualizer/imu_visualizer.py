import rosbag

from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import math
from scipy.signal import butter, lfilter, filtfilt
from scipy.fftpack import fft


class ImuViz():

    def __init__(self, cutoff, ang_cutoff):
        self.time    = []
        self.ang_cmd = []
        self.lin_cmd = []

        self.ang_x   = []
        self.ang_y   = []
        self.ang_z   = []

        self.lin_x   = []
        self.lin_y   = []
        self.lin_z   = []

        self.fs         = 50.0
        self.cutoff     = cutoff
        self.ang_cutoff = ang_cutoff
        
        mpl.style.use('seaborn')


    def load_data(self, filename, cutoff, ang_cutoff, chk_ang, chk_lin):
        print('Loading Data to plot')
        bag = rosbag.Bag('data/' + filename)

        self.time    = []
        self.ang_cmd = []
        self.lin_cmd = []

        self.ang_x   = []
        self.ang_y   = []
        self.ang_z   = []

        self.lin_x   = []
        self.lin_y   = []
        self.lin_z   = []

        self.cutoff     = cutoff
        self.ang_cutoff = ang_cutoff

        for topic, msg, t in bag.read_messages(topics=['cmd_data']):
            self.ang_cmd.append( msg.angular.z )
            self.lin_cmd.append( msg.linear.x  )

        for topic, msg, t in bag.read_messages(topics=['imu_data']):
            self.time.append ( float(str(msg.header.stamp)) )
            self.ang_x.append( msg.angular_velocity.x       )
            self.ang_y.append( msg.angular_velocity.y       )
            self.ang_z.append( msg.angular_velocity.z       )
            self.lin_x.append( msg.linear_acceleration.x    )
            self.lin_y.append( msg.linear_acceleration.y    )
            self.lin_z.append( msg.linear_acceleration.z    )

        bag.close()
        self.x_np = np.array(self.lin_x)
        
        for ii in range(len(self.time)-1):
            self.time[ii+1] = (self.time[ii+1] - self.time[0])/1e9
            
        self.time[0] = 0.0
        if chk_lin:
            self.apply_lin_lowpass()
        if chk_ang:
            self.apply_ang_lowpass()

        print('Loading completed!')

    def apply_lin_lowpass(self):
        order = 6
        nyq = 0.5 * self.fs
        normal_cutoff = self.cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        x = filtfilt(b, a, self.lin_x)
        y = filtfilt(b, a, self.lin_y)
        z = filtfilt(b, a, self.lin_z)
        self.lin_x = x
        self.lin_y = y
        self.lin_z = z

    def apply_ang_lowpass(self):
        order = 6
        nyq = 0.5 * self.fs
        normal_cutoff = self.ang_cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        z = filtfilt(b, a, self.ang_z)
        self.ang_z = z
        y = filtfilt(b, a, self.ang_y)
        self.ang_y = y
        x = filtfilt(b, a, self.ang_x)
        self.ang_x = x

    def running_mean(self, N):
        cumsum = np.cumsum(np.insert(self.x_np, 0, 0))
        self.x_np = (cumsum[N:] - cumsum[:-N]) / N

    def plot_x(self):
        
        #self.lin_x = [value - 0.25 for value in self.lin_x]
        print('Low Pass applied!')
        f0 = plt.figure(1)

        ax1 = plt.subplot2grid((2,2), (0,0))
        ax1.plot(self.time, self.lin_cmd, 'C1')
        ax1.set_title('Linear Command')
        ax1.set_xlabel('time in [s]')
        ax1.set_ylabel('Command Value')

        ax2 = plt.subplot2grid((2,2), (0,1))
        ax2.plot(self.time, self.ang_cmd, 'C2')
        ax2.set_title('Linear Acceleration X')
        ax2.set_xlabel('time in [s]')
        ax2.set_ylabel('Acceleration in [m/s$^2$]')

        ax3 = plt.subplot2grid((2,2), (1,0))
        ax3.plot(self.time, self.lin_x, 'C1')
        ax3.set_title('Linear Acceleration X')
        ax3.set_xlabel('time in [s]')
        ax3.set_ylabel('Command Value')

        ax4 = plt.subplot2grid((2,2), (1,1))
        ax4.plot(self.time, self.ang_x, 'C2')
        ax4.set_title('Angular Velocity X')
        ax4.set_xlabel('time in [s]')
        ax4.set_ylabel('Acceleration in [m/s$^2$]')
        
        
        f0.tight_layout()
      
        return f0
        
    def plot_y(self):
        
        #self.lin_x = [value - 0.25 for value in self.lin_x]
        print('Low Pass applied!')
        f0 = plt.figure(1)

        ax1 = plt.subplot2grid((2,2), (0,0))
        ax1.plot(self.time, self.lin_cmd, 'C1')
        ax1.set_title('Linear Command')
        ax1.set_xlabel('time in [s]')
        ax1.set_ylabel('Command Value')

        ax2 = plt.subplot2grid((2,2), (0,1))
        ax2.plot(self.time, self.ang_cmd, 'C2')
        ax2.set_title('Linear Acceleration X')
        ax2.set_xlabel('time in [s]')
        ax2.set_ylabel('Acceleration in [m/s$^2$]')

        ax3 = plt.subplot2grid((2,2), (1,0))
        ax3.plot(self.time, self.lin_y, 'C1')
        ax3.set_title('Linear Acceleration Y')
        ax3.set_xlabel('time in [s]')
        ax3.set_ylabel('Command Value')

        ax4 = plt.subplot2grid((2,2), (1,1))
        ax4.plot(self.time, self.ang_y, 'C2')
        ax4.set_title('Angular Velocity Y')
        ax4.set_xlabel('time in [s]')
        ax4.set_ylabel('Acceleration in [m/s$^2$]')
        
        
        f0.tight_layout()
      
        return f0

    def plot_z(self):
        
        #self.lin_x = [value - 0.25 for value in self.lin_x]
        print('Low Pass applied!')
        f0 = plt.figure(1)

        ax1 = plt.subplot2grid((2,2), (0,0))
        ax1.plot(self.time, self.lin_cmd, 'C1')
        ax1.set_title('Linear Command')
        ax1.set_xlabel('time in [s]')
        ax1.set_ylabel('Command Value')

        ax2 = plt.subplot2grid((2,2), (0,1))
        ax2.plot(self.time, self.ang_cmd, 'C2')
        ax2.set_title('Linear Acceleration X')
        ax2.set_xlabel('time in [s]')
        ax2.set_ylabel('Acceleration in [m/s$^2$]')

        ax3 = plt.subplot2grid((2,2), (1,0))
        ax3.plot(self.time, self.lin_z, 'C1')
        ax3.set_title('Linear Acceleration Z')
        ax3.set_xlabel('time in [s]')
        ax3.set_ylabel('Command Value')

        ax4 = plt.subplot2grid((2,2), (1,1))
        ax4.plot(self.time, self.ang_z, 'C2')
        ax4.set_title('Angular Velocity Z')
        ax4.set_xlabel('time in [s]')
        ax4.set_ylabel('Acceleration in [m/s$^2$]')
        
        
        f0.tight_layout()
      
        return f0

    def plot_linear(self):
        
        #self.lin_x = [value - 0.25 for value in self.lin_x]
        print('Low Pass applied!')
        f0 = plt.figure(1)

        ax1 = plt.subplot2grid((2,2), (0,0))
        ax1.plot(self.time, self.lin_cmd, 'C1')
        ax1.set_title('Linear Command')
        ax1.set_xlabel('time in [s]')
        ax1.set_ylabel('Command Value')

        ax2 = plt.subplot2grid((2,2), (0,1))
        ax2.plot(self.time, self.lin_x, 'C2')
        ax2.set_title('Linear Acceleration X')
        ax2.set_xlabel('time in [s]')
        ax2.set_ylabel('Acceleration in [m/s$^2$]')

        ax3 = plt.subplot2grid((2,2), (1,0), colspan=2)
        ax3.plot(self.time, self.lin_x, 'C2')
        ax3.plot(self.time, self.lin_cmd, 'C1')
        ax3.set_title('Overlap of Command and Acceleration')
        ax3.set_xlabel('time in [s]')
        ax3.set_ylabel('Velocity in [rad/s]')
        
        f0.tight_layout()
      
        return f0
    def plot_turn(self):
        fig = plt.figure(1)

        ax1 = plt.subplot2grid((2,2), (0,0))
        ax1.plot(self.time, self.ang_cmd, 'C1')
        ax1.set_title('Angular Command')
        ax1.set_xlabel('time in [s]')
        ax1.set_ylabel('Command Value')

        ax2 = plt.subplot2grid((2,2), (0,1))
        ax2.plot(self.time, self.ang_z, 'C2')
        ax2.set_title('Angular Velocity Z')
        ax2.set_xlabel('time in [s]')
        ax2.set_ylabel('Velocity in [rad/s]')

        '''
        ang_cpos = [abs(value) for value in self.ang_cmd]
        ang_ipos = [abs(value) for value in self.ang_z  ]
        
        if max(ang_cpos) == 0.0:
            norm_cmd = len(self.ang_cmd) * [0.0]
        else:
            norm_cmd = [i/max(ang_cpos) for i in self.ang_cmd]
        norm_ang     = [i/max(ang_ipos) for i in self.ang_z  ]
        '''

        ax3 = plt.subplot2grid((2,2), (1,0), colspan=2)
        ax3.plot(self.time, self.ang_z, 'C2')
        ax3.plot(self.time, self.ang_cmd, 'C1')
        ax3.set_title('Overlap of Command and Velocity')
        ax3.set_xlabel('time in [s]')
        ax3.set_ylabel('Velocity in [rad/s]')
        
        fig.tight_layout()

        return fig

    def plot_fft(self):
        N = len(self.lin_x)
        tmp_list = [j-i for i, j in zip(self.time[:-1], self.time[1:])]
        dt = sum(tmp_list)/float(len(tmp_list))
        dt = 0.02

        lFFT = fft(self.lin_x)
        xFFT = np.linspace(0.0, 1.0/(2.0*dt), N)

        fig = plt.figure(1)
        plt.subplot(211)
        plt.plot(self.time, self.lin_x, 'C1')
        plt.xlabel('time in [s]')
        plt.ylabel('Acceleration in [m/s$^2$]')

        plt.subplot(212)
        plt.plot(xFFT, lFFT, 'C2')
        plt.xlabel('frequency in Hz')
        plt.ylabel('FFT value')

        return fig

    def plot_time(self):
        fig = plt.figure(1)

        plt.subplot(111)
        plt.plot(self.time)
        plt.title('Timesteps')
        plt.xlabel('Index')
        plt.ylabel('Time in [s]')

        return fig


    def plot_all(self):

        fig = plt.figure(1)
        
        ax1 = plt.subplot2grid((2,2), (0,0))
        ax1.plot(self.time, self.lin_cmd, 'C1')
        ax1.set_title('Linear Command')
        ax1.set_xlabel('time in [s]')
        ax1.set_ylabel('Command Value')

        ax2 = plt.subplot2grid((2,2), (1,0))
        ax2.plot(self.time, self.lin_x, 'C2')
        ax2.set_title('Linear Acceleration X')
        ax2.set_xlabel('time in [s]')
        ax2.set_ylabel('Acceleration in [m/s$^2$]')

        ax3 = plt.subplot2grid((2,2), (0,1))
        ax3.plot(self.time, self.ang_cmd, 'C3')
        ax3.set_title('Angular Command')
        ax3.set_xlabel('time in [s]')
        ax3.set_ylabel('Command Value')

        ax4 = plt.subplot2grid((2,2), (1,1))
        ax4.plot(self.time, self.ang_z, 'C4')
        ax4.set_title('Angular Velocity Z')
        ax4.set_xlabel('time in [s]')
        ax4.set_ylabel('Velocity in [rad/s]')

        fig.tight_layout()
        

        return fig
