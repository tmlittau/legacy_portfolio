import rospy, rosbag
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu

import os, time, message_filters

class BotSubscriber():

    def __init__(self, filename, imu_topic, cmd_topic):
        rospy.init_node('Imu_analysis', anonymous=True)
        self.fname = filename
        self.imu_sub   = message_filters.Subscriber(imu_topic, Imu)
        self.cmd_sub   = message_filters.Subscriber(cmd_topic, Twist)
        self.imu_data  = []
        self.cmd_data  = []
        self.reference = str(filename)

        ts = message_filters.ApproximateTimeSynchronizer(
            [self.imu_sub, self.cmd_sub], queue_size=20, slop = 0.05, allow_headerless=True)

        ts.registerCallback(self.callback)


    def callback(self, imu_msg, cmd_msg):
        self.imu_data.append(imu_msg)
        self.cmd_data.append(cmd_msg)

    def stop_subscription(self):
        self.imu_sub.unregister()
        self.cmd_sub.unregister()

    def save_data(self):
        bag = rosbag.Bag('data/' + self.fname + '.bag', 'w')
        
        try:
            for ii in range(len(self.imu_data)):
                bag.write('imu_data', self.imu_data[ii])
            for ii in range(len(self.cmd_data)):
                bag.write('cmd_data', self.cmd_data[ii])
        finally:
            bag.close()

        self.imu_data = []
        self.cmd_data = []
