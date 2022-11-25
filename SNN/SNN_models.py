import numpy as np
import matplotlib.pyplot as plt
import norse
from norse.torch.module import Lift
from norse.torch import LICell, LIState, LIF, LIFParameters, LIFState
from norse.torch.functional.lif import LIFParameters
from norse.torch.module.leaky_integrator import LILinearCell
from norse.torch.module.lif import LIFCell, LIFRecurrentCell
from norse.torch.module.conv import LConv2d
from norse.torch.module.leaky_integrator import LI
import torch
from typing import NamedTuple



class SNNState(NamedTuple):
	lif0 : LIFState
	readout : LIState

# Simple SNN from Norse MNIST classification example
class SimpleSNN(torch.nn.Module):
	def __init__(self, input_features, hidden_features, output_features, record=False, dt=0.001):
		super(SNN, self).__init__()
		self.l1 = LIFRecurrentCell(
			input_features,
			hidden_features,
			p=LIFParameters(alpha=100, v_th=torch.tensor(0.5)),
			dt=dt                     
		)
		self.input_features = input_features
		self.fc_out = torch.nn.Linear(hidden_features, output_features, bias=False)
		self.out = LICell(dt=dt)

		self.hidden_features = hidden_features
		self.output_features = output_features
		self.record = record


	def forward(self, x):
		#seq_length, batch_size, _ = x.shape # x: (nb_event, batch_size, 4)
		batch_size, seq_length,  _ = x.shape # x: (batch_size, seq_length,4)
		
		#T = np.max(x[:, :, 3]) # time of the latest event in our data
		T = torch.max(x[:, :, 3]).item()

		s1 = so = None
		voltages = []

		if self.record:
		  self.recording = SNNState(
			  LIFState(
				z = torch.zeros(T, batch_size, self.hidden_features),
				v = torch.zeros(T, batch_size, self.hidden_features),
				i = torch.zeros(T, batch_size, self.hidden_features)
			  ),
			  LIState(
				v = torch.zeros(T, batch_size, self.output_features),
				i = torch.zeros(T, batch_size, self.output_features)
			  )
		  )

		for ts in range(T):
			z = np.zeros((batch_size, self.input_features))

			for i in range(batch_size):
				#events_at_ts_for_batch_i =  np.where(x[:,i,3]==ts)
				#events_at_ts_for_batch_i =  np.where(x[i,:,3]==ts)
				batch_i = x[i]
				#print(batch_i[:20])
				events_at_ts_for_batch_i =  batch_i[batch_i[:,3]==ts]
				#print(events_at_ts_for_batch_i.shape)
				#print(events_at_ts_for_batch_i[0])

				#events_at_ts_for_batch_i = x[events_at_ts_for_batch_i]

				for event in events_at_ts_for_batch_i:
					z[i][event[0]*event[1]]=event[2].item() #Polarity <------ what happens when polarity is 0 ? 
					#print(z[i][event[0]*event[1]])

			z = torch.as_tensor(z).float()

			z, s1 = self.l1(z, s1)
			z = self.fc_out(z)
			vo, so = self.out(z, so)
			if self.record:
			  self.recording.lif0.z[ts,:] = s1.z
			  self.recording.lif0.v[ts,:] = s1.v
			  self.recording.lif0.i[ts,:] = s1.i
			  self.recording.readout.v[ts,:] = so.v
			  self.recording.readout.i[ts,:] = so.i
			voltages += [vo]

		return torch.stack(voltages)

# SNN with convolutions
class ConvNet(torch.nn.Module):
	def __init__(self,  num_channels=1, feature_size_h=500, feature_size_w=650, method="super", alpha=100, num_labels=5):
		super(ConvNet, self).__init__()
		self.feature_size_h = feature_size_h
		self.feature_size_w = feature_size_w
		self.num_channels = num_channels
		self.num_labels=num_labels
		self.features_h = int(((feature_size_h - 4) / 2 - 4) / 2)
		self.features_w = int(((feature_size_w - 4) / 2 - 4) / 2)

		self.conv1 = torch.nn.Conv2d(num_channels, 20, 5, 1)
		self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
		self.fc1 = torch.nn.Linear(self.features_h * self.features_w * 50, 500)

		self.lif0 = LIFCell(p=LIFParameters(method=method, alpha=alpha))
		self.lif1 = LIFCell(p=LIFParameters(method=method, alpha=alpha))
		self.lif2 = LIFCell(p=LIFParameters(method=method, alpha=alpha))
		self.out = LILinearCell(500, num_labels)

	def forward(self, x):
		#print("X tensor size: ", x.element_size() * x.nelement()*0.000001, "MB")
		#print(x.shape)
		seq_length = x.shape[1]
		batch_size = x.shape[0]
		#print("T:", seq_length)
		# specify the initial states
		s0 = s1 = s2 = so = None

		voltages = torch.zeros(
			seq_length, batch_size, self.num_labels, device=x.device, dtype=x.dtype
		)
		#print("Voltage tensor size: ", voltages.element_size() * voltages.nelement()*0.000001, "MB")


		for ts in range(seq_length):
			#print(ts)
			#print("x at ts:", x[:, ts].shape)
			z = torch.reshape(x[:,ts], (x.shape[0], 1, x.shape[2], x.shape[3])) # (batch, channel, w, h)
			z = torch.as_tensor(z).float()
			#print("x at ts:", z.shape)
			#print("z size: ", z.element_size() * z.nelement()*0.000001, "MB")
			z = self.conv1(z)
			#z = self.conv1(z) 
			#print("CONV1: ", z.shape)
			z, s0 = self.lif0(z, s0)
			z = torch.nn.functional.max_pool2d(z, 2, 2)
			#print("POOL1: ", z.shape)
			z = 10 * self.conv2(z)
			#print("CONV2: ", z.shape)
			z, s1 = self.lif1(z, s1)
			z = torch.nn.functional.max_pool2d(z, 2, 2)
			#print("POOL2: ", z.shape)
			z = z.view(-1, z.shape[1]*z.shape[2]*z.shape[3])
			#print(z.shape)
			z = self.fc1(z)      
			#print("fc1 tensor size: ", z.element_size()*z.nelement()*0.000001, "MB")
			#print("FC1: ", z.shape)
			z, s2 = self.lif2(z, s2)
			#print("LIF1: ", z.shape)
			v, so = self.out(torch.nn.functional.relu(z), so)
			#print(v+torch.abs(torch.min(v)))
			voltages[ts, :, :] = v+torch.abs(torch.min(v))
			#print("Voltage tensor size: ", voltages.element_size()*voltages.nelement()*0.000001, "MB")
		return voltages

# Class to wrap SNN model 
class Model(torch.nn.Module):
	def __init__(self, snn, decoder):
		super(Model, self).__init__()
		self.snn = snn
		self.decoder = decoder
		'''
		self.record = snn.record
		self.recording = None
		self.input_features = snn.input_features
		self.hidden_features = snn.hidden_features
		self.output_features = snn.output_features
		self.l1 = snn.l1
		self.fc_out = snn.fc_out
		self.out = snn.out
		'''

	# Here we can decode the outputs of an SNN model and apply a loss function
	def forward(self, x):
		x = self.snn(x)
		log_p_y = self.decoder(x)
		return log_p_y