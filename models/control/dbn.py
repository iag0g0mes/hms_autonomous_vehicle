#import python utils
import numpy as np 
import matplotlib.pyplot as plt

#import smile lib
from smile import pysmile 
from smile import pysmile_licence
from smile.pysmile import SMILEException

from smile.pysmile.learning import EM, DataSet, DataMatch, Validator

class ControlDBN(object):

	def __init__(self, time_slice_frames=3, target_frame=0):

		print ("[Control Diagnostic] running...")

		self.time_slice_frames = time_slice_frames
		self.target_frame = target_frame

		self.net = pysmile.Network()

		self.nodes = {}


	def get_names(self, node):
		if node == 'control':
			names = {'yes':0, 'warn':1, 'no':2}
			return names
		elif 'symptom' in node:
			names = {'very_large':0,
					'large':1,
					'medium':2,
					'small':3,
					'very_small':4,
					'missing':-1}
			return names
		elif 'residue' in node:
			names = {'very_low':0,
					'low_to_very_low':1,
					'low':2,
					'fairly_low':3,
					'medium':4,
					'fairly_high':5,
					'high':6,
					'high_to_very_high':7,
					'very_high':8,
					'missing':-1} 
			return names
		else:
			print ('[Control Diagnostic][Error] node not found!')
			return None

	def get_labels(self, node):
			
		if (node == 'control') or\
			('symptom' in node) or\
			('residue' in node): 

			return {v:k for k,v in self.get_names(node).items()} 

		else:
			print ('[Control Diagnostic][Error] node not found!')
			return None


	def load_network(self, name):
		print ("[Control Diagnostic] loading network ({})...".format(name))

		self.net = pysmile.Network()
		self.nodes = {}
		
		self.net.read_file(name)
		
		for _id in self.net.get_all_nodes():
			_n = self.net.get_node_id(_id)
			self.nodes[_n]= _id

		print ("[Control Diagnostic] nodes:")
		print ('\t\t\t {}'.format(self.nodes))
			

	def save_network(self, name):
		print ("[Control Diagnostic] saving network ({})...".format(name))
		self.net.write_file(name)

	def build(self, name='control_diagnostic_warning.xdsl'):

		self.create_network_discrete()

		self.save_network(name)

	def fit(self, data_file, fixed_nodes=[]):
		print ("[Control Diagnostic][learning] learning from data...")
		print ("[Control Diagnostic][learning] loading dataset ({})...".format(data_file))

		dataset = DataSet()
		dataset.read_file(data_file)

		print ("[Control Diagnostic][learning] matching dataset with network...")

		data_match = DataMatch()

		data_match = dataset.match_network(self.net)

		print ('[Control Diagnostic][learning] learning with EM...')
		em = EM()
		logLik = 0.0

		if len(fixed_nodes) > 0: 
			fixed_nodes_handle = [self.nodes[f] for f in fixed_nodes]
			res = em.learn(dataset, self.net, data_match, fixed_nodes=fixed_nodes_handle)
		else:
			res = em.learn(dataset, self.net, data_match)

		print ('[Control Diagnostic][learning] done!')

	def get_cpt(self, node):
		return np.load('cpt/{}.npy'.format(node))


	def predict(self, observation):

		symptom = np.array([[False, False, False, False]]*self.time_slice_frames)
		residue = np.array([[False, False, False, False]]*self.time_slice_frames)

		residue_names = {'lateral_residue':0,
						 'longitudinal_residue':1,
						 'angular_residue':2,
						 'curvature_residue':3}
		symptom_names = {'lateral_residue':0,
						 'longitudinal_residue':1,
						 'angular_residue':2,
						 'curvature_residue':3}
		fault_names = {'control':0}
		
		for evi in residue_names:
			if (evi in observation) and\
			   (observation[evi] is not None):

			    for temp, value in observation[evi]:
			    	if (value is None) or\
			    	   (np.isnan(value)) or\
			    	   (value<0):
			    	   continue
			    	residue[temp, residue_names[evi]]=True
			    	self.net.set_temporal_evidence(self.nodes[evi], temp, value)

		for evi in symptom_names.keys():
			if (evi in observation) and\
			   (observation[evi] is not None):

			    for temp, value in observation[evi]:
			    	if (value is None) or\
			    	   (np.isnan(value)) or\
			    	   (value<0):
			    	   continue

			    	if residue[temp, :].all():
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

					self.net.set_temporal_evidence(self.nodes[evi], temp, value)


		self.net.update_beliefs()

		result = {}		
		result['residue']  = {}
		result['fault'] = {}
		result['symptom'] = {}


		slice_count = self.net.get_slice_count()
		for h in self.net.get_all_nodes():
			n = self.net.get_node_id(h)

			if 'residue' in n:
				values = self.net.get_node_value(h)
				outcome_count = self.net.get_outcome_count(h)
				nn = n.split('_')[0]

				for slice_idx in range(0, slice_count):
					result['residue']['{}_{}'.format(nn,slice_idx)] = \
							values[slice_idx*outcome_count: (slice_idx+1)*outcome_count]
			elif 'symptom' in n:
				values = self.net.get_node_value(h)
				outcome_count = self.net.get_outcome_count(h)
				nn = n.split('_')[0]

				for slice_idx in range(0, slice_count):
					result['symptom']['{}_{}'.format(nn,slice_idx)] = \
							values[slice_idx*outcome_count: (slice_idx+1)*outcome_count]

			else:
				values = self.net.get_node_value(h)
				outcome_count = self.net.get_outcome_count(h)
				for slice_idx in range(0, slice_count):
					result['fault']['{}_{}'.format(n, slice_idx)] = \
							values[slice_idx*outcome_count: (slice_idx+1)*outcome_count]
		
		return result

	
	def create_cpt_node(self, net, _id, name, outcomes=None):
		handle = net.add_node(pysmile.NodeType.CPT, _id)
		net.set_node_name(handle, name)

		if outcomes is not None:
			initial_outcome_count = net.get_outcome_count(handle)

			for i in range(0, initial_outcome_count):
				net.set_outcome_id(handle, i, outcomes[i])

			for i in range(initial_outcome_count, len(outcomes)):
				net.add_outcome(handle, outcomes[i])

		return handle


	def create_network_discrete(self):

		outcomes_fault = ["yes",
						  "no", 
						  "warn"]
		outcomes_symptom = ["very_large", 
							"large", 
							"medium", 
							"small", 
							"very_small"]
		outcomes_residue = ['very_low',
							'low_to_very_low',
							'low',
							'fairly_low',
							'medium',
							'fairly_high',
							'high',
							'high_to_very_high',
							'very_high']


		#failure node
		control_node = self.create_cpt_node(self.net,
											"control", 
											"Control Diagnostic Node",
											outcomes_fault)
		self.nodes['control']= control_node
		
		#symptoms node
		lat_symptom = self.create_cpt_node(self.net, 
										   "lateral_symptom", 
										   "Lateral Residual Status",
										   outcomes_symptom)
		self.nodes['lateral_symptom'] = lat_symptom

		lon_symptom = self.create_cpt_node(self.net, 
										   "longitudinal_symptom", 
										   "Longitudinal Residual Status",
										   outcomes_symptom)
		self.nodes['longitudinal_symptom'] = lon_symptom

		cur_symptom = self.create_cpt_node(self.net, 
										   "curvature_symptom", 
										   "Kappa Residual Status",
										   outcomes_symptom)
		self.nodes['curvature_symptom'] = cur_symptom

		ang_symptom = self.create_cpt_node(self.net, 
										   "angular_symptom", 
										   "Angular Residual Status",
										   outcomes_symptom)
		self.nodes['angular_symptom'] = ang_symptom

		#evidence node		
		lat_residue = self.create_cpt_node(self.net, 
										   "lateral_residue", 
										   "Lateral Residue", 
											outcomes_residue)
		self.nodes['lateral_residue']= lat_residue

		lon_residue =  self.create_cpt_node(self.net, 
											'longitudinal_residue', 
											'Longitudinal Residue', 
											outcomes_residue)
		self.nodes['longitudinal_residue']= lon_residue


		ang_residue = self.create_cpt_node(self.net, 
											'angular_residue', 
											'Angular Residue', 
											outcomes_residue)
		self.nodes['angular_residue'] = ang_residue


		cur_residue = self.create_cpt_node(self.net, 
										   'curvature_residue', 
										   'Curvature Residue', 
										   outcomes_residue)
		self.nodes['curvature_residue'] = cur_residue

		'''
		contemporal nodes: lat_residue, lon_residue, ang_residue, cur_residue
		plate nodes: lat_symptom, lon_symptom, ang_symptom, cur_symptom, control_res

		temporal_link: control_res_0 -> control_res_1 -> control_res_2
							t-1              t               t+1
		'''
		self.net.set_node_temporal_type(control_node, pysmile.NodeTemporalType.PLATE)
		self.net.set_node_temporal_type(lat_symptom, pysmile.NodeTemporalType.PLATE)
		self.net.set_node_temporal_type(lon_symptom, pysmile.NodeTemporalType.PLATE)
		self.net.set_node_temporal_type(ang_symptom, pysmile.NodeTemporalType.PLATE)
		self.net.set_node_temporal_type(cur_symptom, pysmile.NodeTemporalType.PLATE)
		self.net.set_node_temporal_type(lat_residue, pysmile.NodeTemporalType.PLATE)
		self.net.set_node_temporal_type(lon_residue, pysmile.NodeTemporalType.PLATE)
		self.net.set_node_temporal_type(ang_residue, pysmile.NodeTemporalType.PLATE)
		self.net.set_node_temporal_type(cur_residue, pysmile.NodeTemporalType.PLATE)

		#arcs between temporal slices
		self.net.add_temporal_arc(control_node, control_node, 1)
		self.net.add_temporal_arc(lat_symptom, lat_symptom, 1)
		self.net.add_temporal_arc(lon_symptom, lon_symptom, 1)
		self.net.add_temporal_arc(ang_symptom, ang_symptom, 1)
		self.net.add_temporal_arc(cur_symptom, cur_symptom, 1)

		#arcs between evidence and symptoms
		self.net.add_arc(lat_residue, lat_symptom)
		self.net.add_arc(lon_residue, lon_symptom)
		self.net.add_arc(ang_residue, ang_symptom)
		self.net.add_arc(cur_residue, cur_symptom)

		#arcs between sumptoms and fault
		self.net.add_arc(lat_symptom, control_node)
		self.net.add_arc(lon_symptom, control_node)
		self.net.add_arc(ang_symptom, control_node)
		self.net.add_arc(cur_symptom, control_node)
		
		#target
		self.net.set_target(control_node, True)
		self.net.set_target(lat_residue, True)
		self.net.set_target(lon_residue, True)
		self.net.set_target(ang_residue, True)
		self.net.set_target(cur_residue, True)
		self.net.set_target(lat_symptom, True)
		self.net.set_target(lon_symptom, True)
		self.net.set_target(ang_symptom, True)
		self.net.set_target(cur_symptom, True)

		#number of time slices 
		self.net.set_slice_count(self.time_slice_frames)
