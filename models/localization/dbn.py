from smile import pysmile 
from smile import pysmile_licence
from smile.pysmile.learning import EM, DataSet, DataMatch
import numpy as np

import matplotlib.pyplot as plt



class GPSDBN(object):

	def __init__(self, time_slice_frames=3, target_frame=0):

		print ("[GPS Diagnostic] running...")


		self.time_slice_frames = time_slice_frames
		self.target_frame = target_frame

		self.net = pysmile.Network()

		self.nodes = {}
	


	def predict(self, observation):

		#lat, lon, cur, ang
		symptom  = np.array([[False]*6]*self.time_slice_frames)
		evidence = np.array([[False]*12]*self.time_slice_frames)
		fault    = np.array([[False]]*self.time_slice_frames)

		symptom_names = {'rtk_status_symptom'       :0,
						 'sat_status_symptom'       :1,
						 'pose_residual_symptom'    :2,
						 'pose_variance_symptom'    :3,
						 'pdop_evaluation_symptom'  :4,
						 'raim_status_symptom'      :5}

		evidence_names = {'rtk_evidence'    :0,
						 'sat_evidence'     :1,
						 'innov_x_evidence' :2,
						 'innov_y_evidence' :3,
						 'innov_t_evidence' :4,
						 'var_x_evidence'   :5,
						 'var_y_evidence'   :6,
						 'var_t_evidence'   :7,
						 'pdop_evidence'    :8,
						 'herl_evidence'    :9,
						 'verl_evidence'    :10,
						 'raim_integrity_evidence':11}

		fault_names    = {'gps':0}

		
		for evi in evidence_names.keys():
			if (evi in observation) and\
			   (observation[evi] is not None):

			    for temp, value in observation[evi]:
			    	if (value is None) or\
			    	   (np.isnan(value)) or\
			    	   (value<0):
			    	   continue
			    	evidence[temp, evidence_names[evi]]=True
			    	self.net.set_temporal_evidence(self.nodes[evi], temp, value)

				

		for evi in symptom_names.keys():
			if (evi in observation) and\
			   (observation[evi] is not None):

			    for temp, value in observation[evi]:
			    	if (value is None) or\
			    	   (np.isnan(value)) or\
			    	   (value<0):
			    	   continue

			    	if evidence[temp, :].all():
			    		continue

			    	symptom[temp,symptom_names[evi]] = True
			    	self.net.set_temporal_evidence(self.nodes[evi], temp, value)

		for evi in fault_names.keys():
			if (evi in observation) and\
				(observation[evi] is not None):

				for temp, value in observation[evi]:
					if (value is None) or\
					   (np.isnan(value)) or\
					   (value < 0):
					   continue

					if symptom[temp, :].all():
						continue

					if symptom[temp, :].sum() >=4:
						continue

					fault[temp, fault_names[evi]] = True
					self.net.set_temporal_evidence(self.nodes[evi], temp, value)


		self.net.update_beliefs()

		result = {}		
		result['evidence']  = {}
		result['fault'] = {}
		result['symptom'] = {}


		slice_count = self.net.get_slice_count()
		for h in self.net.get_all_nodes():
			n = self.net.get_node_id(h)

			if 'evidence' in n:
				values = self.net.get_node_value(h)
				outcome_count = self.net.get_outcome_count(h)


				for slice_idx in range(0, slice_count):
					result['evidence']['{}_{}'.format(n,slice_idx)] = \
							values[slice_idx*outcome_count: (slice_idx+1)*outcome_count]
			elif 'symptom' in n:
				values = self.net.get_node_value(h)
				outcome_count = self.net.get_outcome_count(h)


				for slice_idx in range(0, slice_count):
					result['symptom']['{}_{}'.format(n,slice_idx)] = \
							values[slice_idx*outcome_count: (slice_idx+1)*outcome_count]

			else:
				values = self.net.get_node_value(h)
				outcome_count = self.net.get_outcome_count(h)
				for slice_idx in range(0, slice_count):
					result['fault']['{}_{}'.format(n, slice_idx)] = \
							values[slice_idx*outcome_count: (slice_idx+1)*outcome_count]
		
		return result

	def build(self,  name='gps_diagnostic_warning.xdsl'):

		self.create_network_discrete()


		self.save_network(name)


	def fit(self, data_file):
		print ("[GPS Diagnostic][learning] learning from data...")
		print ("[GPS Diagnostic][learning] loading dataset ({})...".format(data_file))

		dataset = DataSet()
		dataset.read_file(data_file)

		print ("[GPS Diagnostic][learning] matching dataset with network...")

		data_match = DataMatch()

		data_match = dataset.match_network(self.net)

		print ('[GPS Diagnostic][learning] learning with EM...')
		em = EM()
		logLik = 0.0
		res = em.learn(dataset, self.net, data_match)

		print ('[GPS Diagnostic][learning] done!')	
	

	def create_network_discrete(self):

		outcomes = {}
		outcomes['fault'] = ["yes", "no"] 
		
		outcomes['symptom'] = {}
		outcomes['symptom']['rtk'] = ["good", "medium", "bad", "very_bad", "no_correction"] 
		outcomes['symptom']['dop'] = ["ideal", "excellent", "good", "moderate", "fair", "poor", "not_available"]
		outcomes['symptom']['satellites']    = ["not_enough",  "enough", "good"]
		outcomes['symptom']['pose_residue']  =  ['very_large','large', 'medium', 'small', 'very_small']
		outcomes['symptom']['pose_variance'] =  ['very_large','large', 'medium', 'small', 'very_small']
		outcomes['symptom']['raim'] =  ["yes",  "warn", "no"]

		outcomes['evidence'] = {}
		outcomes['evidence']['raim_integrity'] = ["successful", "failed", "not_available"]
		outcomes['evidence']['innov'] = ['very_low', 'low_to_very_low',
										 'low', 'fairly_low',
										 'medium','fairly_high',
										 'high', 'high_to_very_high',
										 'very_high']
		outcomes['evidence']['var']   = ['very_low', 'low_to_very_low',
										 'low', 'fairly_low',
										 'medium','fairly_high',
										 'high', 'high_to_very_high',
										 'very_high']
		outcomes['evidence']['rtk']   = ['very_low', 'low_to_very_low',
										 'low', 'fairly_low',
										 'medium','fairly_high',
										 'high', 'high_to_very_high',
										 'very_high']
		outcomes['evidence']['sat']   = ['very_low', 'low_to_very_low',
										 'low', 'fairly_low',
										 'medium','fairly_high',
										 'high', 'high_to_very_high',
										 'very_high']

		outcomes['evidence']['pdop']  =	['very_low', 'low',
										 'fairly_low', 'medium',
										 'fairly_high', 'high',
										 'very_high']

		outcomes['evidence']['herl_verl']  = ["yes", "no"]

		#failure nodes
		gps_node = self.create_cpt_node(self.net,
										"gps", 
										"GPS Diagnostic Node",
										outcomes['fault'])
		self.nodes['gps']= gps_node

		#symptoms node
		rtk_symptom = self.create_cpt_node(self.net, 
										   "rtk_status_symptom", 
										   "Status of RTK Correction",
											outcomes['symptom']['rtk'])
		self.nodes['rtk_status_symptom']= rtk_symptom

		pdop_symptom = self.create_cpt_node(self.net, 
											"pdop_evaluation_symptom", 
											"PDOP Evaluation",
											outcomes['symptom']['dop'])
		self.nodes['pdop_evaluation_symptom'] = pdop_symptom

		sat_symptom = self.create_cpt_node(self.net, 
										   "sat_status_symptom", 
										   "Status of Num. of Sattelites",
											outcomes['symptom']['satellites'])
		self.nodes['sat_status_symptom'] = sat_symptom

		posr_symptom=self.create_cpt_node(self.net,
										  "pose_residual_symptom", 
										  "Pose Residual",
										   outcomes['symptom']['pose_residue'])
		self.nodes['pose_residual_symptom']= posr_symptom	

		posv_symptom=self.create_cpt_node(self.net,
										  "pose_variance_symptom", 
										  "Pose Variance",
										  outcomes['symptom']['pose_variance'])
		self.nodes['pose_variance_symptom'] = posv_symptom


		raim_symptom = self.create_cpt_node(self.net, 
											"raim_status_symptom", 
											"RAIM algorithm status",
											outcomes['symptom']['raim'])
		self.nodes['raim_status_symptom'] = raim_symptom


		#evidence node
		rtk_evidence = self.create_cpt_node(self.net, 
											"rtk_evidence", 
											"RTK correction age", 
											outcomes['evidence']['rtk'])
		self.nodes['rtk_evidence']= rtk_evidence

		sat_evidence = self.create_cpt_node(self.net, 
										"sat_evidence", 
										"Number of Sattelites", 
										outcomes['evidence']['sat'])
		self.nodes['sat_evidence']= sat_evidence

		innov_x_evidence=self.create_cpt_node(self.net,
											 "innov_x_evidence", 
											 "Innovation on x position", 
											  outcomes['evidence']['innov'])
		self.nodes['innov_x_evidence']= innov_x_evidence		

		innov_y_evidence=self.create_cpt_node(self.net,
											  "innov_y_evidence", 
											  "Innovation on y position", 
											  outcomes['evidence']['innov'])
		self.nodes['innov_y_evidence']= innov_y_evidence	

		innov_t_evidence=self.create_cpt_node(self.net,
											  "innov_t_evidence", 
											  "Innovation on yaw angle", 
											  outcomes['evidence']['innov'])
		self.nodes['innov_t_evidence']= innov_t_evidence	
		
		var_x_evidence=self.create_cpt_node(self.net,
											"var_x_evidence", 
											"Variance on x position", 
											 outcomes['evidence']['var'])
		self.nodes['var_x_evidence']= var_x_evidence		

		var_y_evidence=self.create_cpt_node(self.net,
											"var_y_evidence", 
											"Variance on y position", 
											 outcomes['evidence']['var'])
		self.nodes['var_y_evidence']= var_y_evidence	

		var_t_evidence=self.create_cpt_node(self.net,
											"var_t_evidence", 
											"Variance on yaw angle", 
											 outcomes['evidence']['var'])
		self.nodes['var_t_evidence']= var_t_evidence	

		 
		pdop_evidence=self.create_cpt_node(self.net,
										   "pdop_evidence", 
										   "PDOP", 
										   outcomes['evidence']['pdop'])
		self.nodes['pdop_evidence']= pdop_evidence

		raim_integrity_evidence = self.create_cpt_node(self.net, 
													   "raim_integrity_evidence", 
													   "RAIM Integrity",
													   outcomes['evidence']['raim_integrity'])
		self.nodes['raim_integrity_evidence'] = raim_integrity_evidence

		herl_evidence = self.create_cpt_node(self.net,
											 "herl_evidence", 
											 "HERL Position", 
											 outcomes['evidence']['herl_verl'])
		self.nodes['herl_evidence']=herl_evidence

		verl_evidence = self.create_cpt_node(self.net,
											"verl_evidence", 
											"VERL Position", 
											outcomes['evidence']['herl_verl'])
		self.nodes['verl_evidence'] = verl_evidence


		#set temporal nodes
	  
		self.net.set_node_temporal_type(gps_node, pysmile.NodeTemporalType.PLATE)

		self.net.set_node_temporal_type(rtk_symptom, pysmile.NodeTemporalType.PLATE)
		self.net.set_node_temporal_type(pdop_symptom, pysmile.NodeTemporalType.PLATE)
		self.net.set_node_temporal_type(sat_symptom, pysmile.NodeTemporalType.PLATE)
		self.net.set_node_temporal_type(posr_symptom, pysmile.NodeTemporalType.PLATE)
		self.net.set_node_temporal_type(posv_symptom, pysmile.NodeTemporalType.PLATE)
		self.net.set_node_temporal_type(raim_symptom, pysmile.NodeTemporalType.PLATE)

		self.net.set_node_temporal_type(rtk_evidence, pysmile.NodeTemporalType.PLATE)
		self.net.set_node_temporal_type(sat_evidence, pysmile.NodeTemporalType.PLATE)
		self.net.set_node_temporal_type(innov_x_evidence, pysmile.NodeTemporalType.PLATE)
		self.net.set_node_temporal_type(innov_y_evidence, pysmile.NodeTemporalType.PLATE)
		self.net.set_node_temporal_type(innov_t_evidence, pysmile.NodeTemporalType.PLATE)
		self.net.set_node_temporal_type(var_x_evidence, pysmile.NodeTemporalType.PLATE)
		self.net.set_node_temporal_type(var_y_evidence, pysmile.NodeTemporalType.PLATE)
		self.net.set_node_temporal_type(var_t_evidence, pysmile.NodeTemporalType.PLATE)
		self.net.set_node_temporal_type(pdop_evidence, pysmile.NodeTemporalType.PLATE)
		self.net.set_node_temporal_type(raim_integrity_evidence, pysmile.NodeTemporalType.PLATE)
		self.net.set_node_temporal_type(herl_evidence, pysmile.NodeTemporalType.PLATE)
		self.net.set_node_temporal_type(verl_evidence, pysmile.NodeTemporalType.PLATE)


		#arcs between temporal slices
		self.net.add_temporal_arc(gps_node, gps_node, 1)

		self.net.add_temporal_arc(rtk_symptom,  rtk_symptom,  1)
		self.net.add_temporal_arc(pdop_symptom, pdop_symptom, 1)
		self.net.add_temporal_arc(sat_symptom,  sat_symptom,  1)
		self.net.add_temporal_arc(posr_symptom, posr_symptom, 1)
		self.net.add_temporal_arc(posv_symptom, posv_symptom, 1)
		self.net.add_temporal_arc(raim_symptom, raim_symptom, 1)


		#arcs between symptons and evidences
		self.net.add_arc(innov_x_evidence, posr_symptom)
		self.net.add_arc(innov_y_evidence, posr_symptom)
		self.net.add_arc(innov_t_evidence, posr_symptom)
		
		self.net.add_arc(var_x_evidence, posv_symptom)
		self.net.add_arc(var_y_evidence, posv_symptom)
		self.net.add_arc(var_t_evidence, posv_symptom)
		
		self.net.add_arc(rtk_evidence, rtk_symptom)

		self.net.add_arc(sat_evidence, sat_symptom)

		self.net.add_arc(herl_evidence, raim_symptom)
		self.net.add_arc(verl_evidence, raim_symptom)
		self.net.add_arc(raim_integrity_evidence, raim_symptom)

		self.net.add_arc(pdop_evidence, pdop_symptom)

		#arcs between symptons and fault
		self.net.add_arc(raim_symptom,  gps_node)
		self.net.add_arc(pdop_symptom, gps_node)
		self.net.add_arc(sat_symptom,  gps_node)
		self.net.add_arc(rtk_symptom,  gps_node)
		self.net.add_arc(posr_symptom, gps_node)
		self.net.add_arc(posv_symptom, gps_node)
		
		#target diagnostic		
		self.net.set_target(gps_node, True)
		self.net.set_target(raim_symptom, True)
		self.net.set_target(pdop_symptom, True)
		self.net.set_target(sat_symptom,  True)
		self.net.set_target(rtk_symptom,  True)
		self.net.set_target(posr_symptom, True)
		self.net.set_target(posv_symptom, True)
		self.net.set_target(innov_x_evidence, True)
		self.net.set_target(innov_y_evidence, True)
		self.net.set_target(innov_t_evidence, True)
		self.net.set_target(var_x_evidence, True)
		self.net.set_target(var_y_evidence, True)
		self.net.set_target(var_t_evidence, True)
		self.net.set_target(rtk_evidence, True)
		self.net.set_target(sat_evidence, True)
		self.net.set_target(herl_evidence, True)
		self.net.set_target(verl_evidence, True)
		self.net.set_target(raim_integrity_evidence, True)
		self.net.set_target(pdop_evidence, True)


		#number of time slices 
		self.net.set_slice_count(self.time_slice_frames)



	def create_cpt_node(self, net, id, name, outcomes=None):
		handle = net.add_node(pysmile.NodeType.CPT, id)
		net.set_node_name(handle, name)
		if outcomes is not None:
			initial_outcome_count = net.get_outcome_count(handle)
			for i in range(0, initial_outcome_count):
				net.set_outcome_id(handle, i, outcomes[i])
			for i in range(initial_outcome_count, len(outcomes)):
				net.add_outcome(handle, outcomes[i])
		return handle
		
	def load_network(self, name):
		print ("[GPS Diagnostic] loading network ({})...".format(name))
		self.net.read_file(name)
		self.nodes = {}
		
		for _id in self.net.get_all_nodes():
			_n = self.net.get_node_id(_id)
			self.nodes[_n]= _id

		print ("[GPS Diagnostic] nodes:")
		print ('\t\t\t{}'.format(self.nodes))
			

	def save_network(self, name):
		print ("[GPS Diagnostic] saving network ({})...".format(name))
		self.net.write_file(name)
