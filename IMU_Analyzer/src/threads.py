from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QByteArray
import subprocess
import sys
import os

from src.NN.NN import LSTM
from src.Synth.data_synth import Synthesizer

class synth_Thread(QThread):
    
    signal = pyqtSignal('PyQt_PyObject')

    def __init__(self, fname, t_len, window):
        QThread.__init__(self)
        self.fname      = fname
        self.timelen    = t_len
        self.mainWindow = window

    def run(self):
        synth = Synthesizer(self.timelen, self.mainWindow)
        synth.synth_loop(self.mainWindow)
        synth.save_synth(self.fname, self.mainWindow)
        newfile = self.fname + '.bag'
        self.mainWindow.fileBox.addItem(newfile)
        self.mainWindow.train_box.addItem(newfile)
        del synth
        sig = '[Synthesizer] Done!'
        self.signal.emit(sig)

class NN_Thread(QThread):
    
    signal = pyqtSignal('PyQt_PyObject')

    def __init__(self, fname, mname, window):
        QThread.__init__(self)
        self.fname      = fname
        self.mname      = mname
        self.mainwindow = window
        self.train      = True

    def run(self):
        self.mainwindow.write_to_terminal('[Model] Initializing Data for Training')
        my_model = LSTM(self.fname)
        if self.train:
            print('running training now...')
            my_model.training(self.mainwindow)
            mname = my_model.testing(self.mainwindow)
            self.mainwindow.modelBox.addItem(mname)
            strang = '[Model] Done!'
            self.signal.emit(strang)
        else:
            my_model.model.load_model(self.mname, self.mainwindow)
            my_model.testing_whole(self.mainwindow)
            strang = '[Model] Done!'
            self.signal.emit(strang)
        del my_model

class MoviePlayer(QtWidgets.QWidget): 
    def __init__(self, parent=None): 
        QtWidgets.QWidget.__init__(self, parent) 
        # setGeometry(x_pos, y_pos, width, height)
        self.setGeometry(200, 200, 830, 700)
        self.setWindowTitle("Loading")
        
        # set up the movie screen on a label
        self.movie_screen = QtWidgets.QLabel()
        # expand and center the label 
        self.movie_screen.setSizePolicy(QtWidgets.QSizePolicy.Expanding, 
            QtWidgets.QSizePolicy.Expanding)        
        self.movie_screen.setAlignment(Qt.AlignCenter)        
        main_layout = QtWidgets.QVBoxLayout() 
        main_layout.addWidget(self.movie_screen)
        self.setLayout(main_layout) 
                
        # use an animated gif file you have in the working folder
        # or give the full file path
        self.movie = QtGui.QMovie("loading.gif", QByteArray(), self) 
        self.movie.setCacheMode(QtGui.QMovie.CacheAll) 
        self.movie.setSpeed(100) 
        self.movie_screen.setMovie(self.movie) 
        
    def start(self):
        """sart animnation"""
        self.movie.start()
        
    def stop(self):
        """stop the animation"""
        self.movie.stop()  
