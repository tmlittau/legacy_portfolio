from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QByteArray
import subprocess
import sys
import os
import src.design as design
import json
import time
import rospy

from src.Collector.IMU_subscriber import BotSubscriber
from src.Visualizer.imu_visualizer import ImuViz
from src.NN.NN import LSTM
from src.Synth.data_synth import Synthesizer

from src.threads import NN_Thread, synth_Thread

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)

class imuAnalyzer(QtWidgets.QMainWindow, design.Ui_MainWindow):

    def __init__(self, parent=None):
        # Initialization of the main window
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.setWindowIcon(QtGui.QIcon('icon.png'))

        # Accessing the all the files in subfolders to fill comboBoxes
        filelist  = os.listdir('data')
        modelList = os.listdir('saved_models') 
        self.fileBox.addItems(filelist)
        self.train_box.addItems(filelist)
        self.modelBox.addItems(modelList)

        # Initializing Progressbars and filenames with the random inputs
        self.progressBar.setTextVisible(False)
        self.epoch_progress.setTextVisible(False)
        self.synth_progress.setTextVisible(False)
        self.fname_stream = self.line_fname.text()
        self.fname        = self.fileBox.currentText()
        self.fname_train  = self.train_box.currentText()
        self.mname        = self.modelBox.currentText()

        # Initializing generic settings of GUI
        self.imu_topic    = self.line_imutopic.text()
        self.cmd_topic    = self.line_cmdtopic.text()
        self.runtime      = self.runtime_box.value()
        
        self.vizsetting   = self.vizSet_box.currentText()
        self.cutoff       = self.cutoff_box.value()
        self.ang_cutoff   = self.cutoff_angBox.value()

        # Initializing the NN Model Object and Vizualizer Object
        self.my_model     = LSTM(self.fname_train)
        self.Vizard       = ImuViz(self.cutoff, self.ang_cutoff)

        # Initializing the Threads used in different occasions
        self.model_thread = NN_Thread(self.fname_train, self.mname, self)
        self.model_thread.signal.connect(self.prediction_finished)

        self.synth_thread = synth_Thread(self.synthFile_line.text(),
                                         self.synthOption_box.currentText(),
                                         self)
        self.synth_thread.signal.connect(self.synth_finished)
        
        # Initializing Figures
        self.fig          = Figure()
        self.addmpl()

        # connecting buttons to their methods
        self.btn_start.clicked.connect(self.start_collection)
        self.btn_viz.clicked.connect(self.visualize)
        self.btn_train.clicked.connect(self.start_train)
        self.btn_model.clicked.connect(self.load_and_predict)
        self.btn_gen.clicked.connect(self.synthesize_data)

        

    def write_to_terminal(self, string):
        self.terminal.addItem(string)
    def write_to_synth(self, string):
        self.synth_terminal.addItem(string)

    def init_progbar(self, maxval):
        self.epoch_progress.setMaximum(maxval)  
    def update_progbar(self, current):
        self.epoch_progress.setValue(current)
    def init_sbar(self, maxval):
        self.synth_progress.setMaximum(maxval)
    def update_sbar(self, current):
        self.synth_progress.setValue(current)

    def start_collection(self):
        self.fname_stream = self.line_fname.text()
        self.imu_topic    = self.line_imutopic.text()
        self.cmd_topic    = self.line_cmdtopic.text()
        self.runtime      = self.runtime_box.value()
        
        #rospy.init_node('Imu_analysis', anonymous=True)

        self.progressBar.setMaximum(self.runtime)
        
        if self.runtime == 0:
            self.progressBar.setMaximum(60)
            
        self.progressBar.setValue(0)
        elapsed = 0.0
        
        if self.runtime > 0:
            t_start = time.time()
            subscriber = BotSubscriber(self.fname_stream, self.imu_topic, self.cmd_topic)
            while elapsed <= self.runtime:
                elapsed = time.time() - t_start
                self.progressBar.setValue(elapsed)
            subscriber.stop_subscription()
            subscriber.save_data()
            self.fileBox.addItem(str(self.fname_stream) + '.bag')
            print('DONE!')
        else:
            print('ERROR: unlimited time not yet implemented')        

    def visualize(self):
        self.rmmpl()
        self.vizsetting = self.vizSet_box.currentText()
        if (self.fname  != self.fileBox.currentText() or
            self.cutoff != self.cutoff_box.value() or
            self.ang_cutoff != self.cutoff_angBox.value()):
            self.cutoff     = self.cutoff_box.value()
            self.ang_cutoff = self.cutoff_angBox.value()
            self.fname      = self.fileBox.currentText()
            chk_ang         = self.angfilBox.isChecked()
            chk_lin         = self.linfilBox.isChecked()
            self.Vizard.load_data(self.fname, self.cutoff, self.ang_cutoff, chk_ang, chk_lin)
        
        if self.vizsetting   == 'Linear':
            fig1 = self.Vizard.plot_linear()
        elif self.vizsetting == 'Angular':
            fig1 = self.Vizard.plot_turn()
        elif self.vizsetting == 'All':
            fig1 = self.Vizard.plot_all()
        elif self.vizsetting == 'Time':
            fig1 = self.Vizard.plot_time()
        elif self.vizsetting == 'FFT':
            fig1 = self.Vizard.plot_fft()
        elif self.vizsetting == 'X':
            fig1 = self.Vizard.plot_x()
        elif self.vizsetting == 'Y':
            fig1 = self.Vizard.plot_y()
        elif self.vizsetting == 'Z':
            fig1 = self.Vizard.plot_z()
        

        self.fig = fig1
        self.addmpl()

    def start_train(self):
        self.model_thread.fname = self.train_box.currentText()
        self.model_thread.train = True
        self.btn_train.setEnabled(False)
        self.btn_model.setEnabled(False)
        self.model_thread.start()

    def load_and_predict(self):
        self.model_thread.fname = self.train_box.currentText()
        self.model_thread.mname = self.modelBox.currentText()
        self.model_thread.train = False
        self.btn_train.setEnabled(False)
        self.btn_model.setEnabled(False)
        self.model_thread.start()

    def prediction_finished(self, result):
        strang = result
        self.write_to_terminal(strang)
        self.btn_train.setEnabled(True)
        self.btn_model.setEnabled(True)
        self.model_thread.quit()
        
    def synthesize_data(self):
        self.synth_thread.timelen = self.synthOption_box.currentText()
        self.synth_thread.fname   = self.synthFile_line.text()
        self.btn_gen.setEnabled(False)
        self.synth_thread.start()

    def synth_finished(self, result):
        sig = result
        print(sig)
        self.btn_gen.setEnabled(True)
        
    def addmpl(self):
        self.canvas = FigureCanvas(self.fig)
        self.mplvl.addWidget(self.canvas)
        self.canvas.draw()
        self.toolbar = NavigationToolbar(self.canvas, 
                self.mplwindow, coordinates=True)
        self.mplvl.addWidget(self.toolbar)
        
    def rmmpl(self,):
        self.fig.clf()
        self.mplvl.removeWidget(self.canvas)
        self.canvas.close()
        self.mplvl.removeWidget(self.toolbar)
        self.toolbar.close()

def main():
    app = QtWidgets.QApplication(sys.argv)
    form = imuAnalyzer()
    form.show()
    app.exec_()


if __name__== '__main__':
    main()
