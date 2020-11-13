
#import python utils
import numpy as np 
import matplotlib.pyplot as plt
import time
import os
import json
import pickle
import pandas as pd
from tqdm import tqdm

#import smile lib
from smile import pysmile 
from smile import pysmile_licence


#import model
from dbn import GPSDBN



if __name__ == '__main__':

	print ('Init Localization BN...')

	count_slices = 11
	target_frame = count_slices//2

	model  = GPSDBN(time_slice_frames=count_slices, target_frame=target_frame)

	#build
	# model.build(name='gps_diagnosis_discrete.xdsl')

	#learning
	# model.load_network(name='gps_diagnosis_discrete.xdsl')
	# model.fit(data_file='datafile_train.csv')
	# model.save_network(name='gps_diagnosis_discrete.xdsl')

	#evaluate
	model.load_network(name='dbn_localization.xdsl')








