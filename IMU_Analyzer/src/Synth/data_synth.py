import os
import csv
import math
import random
import json

import rosbag
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.style.use('seaborn')


class Synthesizer():

    def __init__(self, timelength, window):
        self.configs = json.load(open('synth_config.json', 'r'))

        self.dt = 1./self.configs['signal_rate']
        self.t_len = float(timelength)
        self.delay = self.configs['delay']

        self.time = time = np.arange(0.0, (self.t_len + self.dt), self.dt, dtype=np.float)

        self.imu_data = np.zeros((self.time.size, 6))
        self.cmd_data = np.zeros((self.time.size, 2))

        self.synth_type = self.configs['cycle_type']

        self.count = 0.0
        maxval = int(self.t_len/self.dt)*3+self.time.size-int(self.delay/self.dt)
        window.init_sbar(maxval)
        

    def gen_time(self):
        time = np.arange(0.0, (self.t_len + self.dt), self.dt, dtype=np.float)
        return time

    def synth_loop(self, window):
        print('starting synthetic data generation...')
        window.write_to_synth('[Synthesizer] Starting Data Generation...')
        tstep = 0
        i_delay = int(self.delay/self.dt)
        while tstep < int(self.t_len/self.dt):
            vx_cmd = random.randint(-5,5)/10.
            if tstep == 0:
                vx_cmd = 0.0
            vx_dur = min(random.randint(10,500), int((self.t_len/self.dt)-tstep))
            jj = 1
            speed = 0.002
            steps = min(abs(int((vx_cmd - self.cmd_data[tstep,0])/speed)), vx_dur)
            speed = np.sign(vx_cmd - self.cmd_data[tstep,0])*speed
            while jj <= steps:
                self.cmd_data[tstep+jj,0] = self.cmd_data[tstep+jj-1,0] + speed
                jj += 1
            if steps != vx_dur:
                self.cmd_data[tstep+steps:tstep+vx_dur+1,0] = vx_cmd
            tstep +=vx_dur
            self.count += vx_dur
            window.update_sbar(self.count)

        tstep = 0
        while tstep < int(self.t_len/self.dt):
                
            vz_cmd = random.randint(-10,10)/10.
            if tstep == 0:
                vz_cmd = 0.0
            vz_dur = min(random.randint(10,500),int((self.t_len/self.dt)-tstep))
            #dur = max(vz_dur, vx_dur)
            jj = 1
            speed = 0.01
            steps = min(abs(int((vz_cmd - self.cmd_data[tstep,1])/speed)), vz_dur)
            speed = np.sign(vz_cmd - self.cmd_data[tstep,1]) * speed
            while jj <= steps:
                self.cmd_data[tstep+jj,1] = self.cmd_data[tstep+jj-1,1] + speed
                jj += 1

            if steps != vz_dur:
                self.cmd_data[tstep+steps:tstep+vz_dur+1,1] = vz_cmd

            tstep += vz_dur
            self.count += vz_dur
            window.update_sbar(self.count)
        print('Commando Series generated!')
        window.write_to_synth('[Synthesizer] Commando Series generated!')
        print('starting IMU data generation')
        window.write_to_synth('[Synthesizer] Starting IMU data generation')
        ddtCMD = []
        ddtCMD.append(0.0)

        for tt in range(len(self.cmd_data[:,0])-2):
            ddtCMD.append((self.cmd_data[tt+2,0] - self.cmd_data[tt,0])/(2.*self.dt))
                
        ddtCMD.append(0.0)

        for tt in range(len(self.cmd_data[:,0]) - i_delay):
            ang_noise_z = (random.random()- 0.5)*4.
            lin_noise_x = (random.random()- 0.5)/2.

            vf_ang_z = self.cmd_data[tt,1] * 50. + ang_noise_z
            vf_lin_x = float(self.cmd_data[tt,1] != 0)*2.*lin_noise_x + lin_noise_x*float(self.cmd_data[tt,1] == 0)

            self.imu_data[tt+i_delay,0] = vf_lin_x
            self.imu_data[tt+i_delay,5] = vf_ang_z
            
            self.imu_data[tt+i_delay, 0] = self.imu_data[tt+i_delay, 0] - ddtCMD[tt]*5.
            self.count += 1
            window.update_sbar(self.count)
        print('Data Generation finished!')
        window.write_to_synth('[Synthesizer] Data Synthesis completed')
              
    def plot_synthdata(self):

        print('Plotting results...')
        fig = plt.figure(1)
        
        plt.subplot(221)
        plt.plot(self.time[0:3000], self.cmd_data[0:3000,1], 'C1')
        plt.title('Angular Command')
        plt.xlabel('time in [s]')
        plt.ylabel('Command Value')

        plt.subplot(222)
        plt.plot(self.time[0:3000], self.cmd_data[0:3000,0], 'C3')
        plt.title('Linear Command')
        plt.xlabel('time in [s]')
        plt.ylabel('Command Value')

        plt.subplot(223)
        plt.plot(self.time[0:3000], self.imu_data[0:3000,5], 'C4')
        plt.title('Angular Velocity Z')
        plt.xlabel('time in [s]')
        plt.ylabel('Velocity in [m/s]')
        
        plt.subplot(224)
        plt.plot(self.time[0:3000], self.imu_data[0:3000,0], 'C2')
        plt.title('Linear Acceleration X')
        plt.xlabel('time in [s]')
        plt.ylabel('Acceleration in [m/s$^2$]')        

        fig.tight_layout()
        

        return fig
        

    def save_synth(self, fname, window):
        print('Saving Bagfile...')
        window.write_to_synth('[Synthesizer] Saving Bagfile')
        bag = rosbag.Bag('data/' + fname + '.bag', 'w')
        
        imu_msgs = []
        cmd_msgs = []
        
        imu_msg = Imu()
        cmd_msg = Twist()
        try:
            for ii in range(int(self.t_len/self.dt)):
                imu_msg.linear_acceleration.x = self.imu_data[ii,0]
                imu_msg.linear_acceleration.y = self.imu_data[ii,1]
                imu_msg.linear_acceleration.z = self.imu_data[ii,2]
                imu_msg.angular_velocity.x    = self.imu_data[ii,3]
                imu_msg.angular_velocity.y    = self.imu_data[ii,4]
                imu_msg.angular_velocity.z    = self.imu_data[ii,5]
                imu_msg.header.stamp          = rospy.Time(self.time[ii])
                
                cmd_msg.linear.x              = self.cmd_data[ii,0]
                cmd_msg.angular.z             = self.cmd_data[ii,1]
                
                imu_msgs.append(imu_msg)
                cmd_msgs.append(cmd_msg)

                bag.write('imu_data', imu_msgs[ii])
                bag.write('cmd_data', cmd_msgs[ii])
                self.count += 1
                window.update_sbar(self.count)
        finally:
            bag.close()


        print('everything Done!')
        window.write_to_synth('[Synthesizer] Done!')

        def __del__(self):
            self.cleanup()
            
            print('Synthesizer Object and allocated Memory cleaned')
