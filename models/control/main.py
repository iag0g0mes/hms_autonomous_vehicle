
#import python utils
import numpy as np 
import matplotlib.pyplot as plt
import time
import os
import json
import pickle
import pandas as pd

#import smile lib
from smile import pysmile 
from smile import pysmile_licence


#import model
from dbn import ControlDBN


if __name__ == '__main__':

	print ('Init Control BN...')

	count_slices = 11
	target_frame = count_slices//2
	model  = ControlDBN(time_slice_frames=count_slices, target_frame=target_frame)

	#build
	# model.build(name='control_diagnosis_discrete.xdsl')

	#load
	# model.load_network(name='control_diagnosis_discrete.xdsl')

	# #learning
	# model.load_network(name='control_diagnosis_discrete.xdsl')
	# model.fit(data_file='datafile_train.csv')
	# model.save_network(name='control_diagnosis_discrete.xdsl')

	model.load_network(name='dbn_control.xdsl')
