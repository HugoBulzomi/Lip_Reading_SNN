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
		print(self.features_h)
		print(self.features_w)
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
			print(z.shape)
			z = self.fc1(z)      
			#print("fc1 tensor size: ", z.element_size()*z.nelement()*0.000001, "MB")
			print("FC1: ", z.shape)
			z, s2 = self.lif2(z, s2)
			#print("LIF1: ", z.shape)
			v, so = self.out(torch.nn.functional.relu(z), so)
			#print(v+torch.abs(torch.min(v)))
			voltages[ts, :, :] = v+torch.abs(torch.min(v))
			#print("Voltage tensor size: ", voltages.element_size()*voltages.nelement()*0.000001, "MB")
		return voltages



# Smaller version of TrueNorth model, with only the first 8 convolution layers (as described in SLAYER ?)
class TrueNorthSmall(torch.nn.Module):
	def __init__(self,  num_channels=1, feature_size_h=500, feature_size_w=650, method="super", alpha=100, num_labels=5):
		super(TrueNorthSmall, self).__init__()
		self.feature_size_h = feature_size_h
		self.feature_size_w = feature_size_w
		self.num_channels = num_channels
		self.num_labels=num_labels

		self.features_h = int( (((feature_size_h - 4)/2-4) /2-4)/ 2)
		self.features_w = int( (((feature_size_w - 4)/2-4) /2-4)/ 2)
		print(self.features_h)
		print(self.features_w)

		# TrueNorth paper describes binary activation functions for convolutional neurons and trinary weights {-1, 0, 1} -> how to do that in pytorch ?
		self.conv1 = torch.nn.Conv2d(num_channels, 12, 5, 1)
		self.lif1 = LIFCell(p=LIFParameters(method=method, alpha=alpha))
		self.maxpool1 = torch.nn.MaxPool2d(2, 2)

		self.conv2 = torch.nn.Conv2d(12, 24, 5, 1)
		self.conv3 = torch.nn.Conv2d(24, 24, 5, 1)
		self.lif2 = LIFCell(p=LIFParameters(method=method, alpha=alpha))
		self.maxpool2 = torch.nn.MaxPool2d(2, 2)

		self.conv4 = torch.nn.Conv2d(24, 48, 5, 1)
		self.conv5 = torch.nn.Conv2d(48, 48, 5, 1)
		self.conv6 = torch.nn.Conv2d(48, 48, 5, 1)
		self.conv7 = torch.nn.Conv2d(48, 48, 5, 1)
		self.conv8 = torch.nn.Conv2d(48, 48, 5, 1)
		self.lif3 = LIFCell(p=LIFParameters(method=method, alpha=alpha))
		self.maxpool3 = torch.nn.MaxPool2d(2, 2)

		self.fc1 = torch.nn.Linear(24192, 500)
		self.lif4 = LIFCell(p=LIFParameters(method=method, alpha=alpha))
		self.out = LILinearCell(500, num_labels) # Leaky Linear Cells as output -> Use spiking neuron WTA instead ?

	def forward(self, x):
		seq_length = x.shape[1]
		batch_size = x.shape[0]
		s1 = s2 = s3 = s4 = so = None

		voltages = torch.zeros(
			seq_length, batch_size, self.num_labels, device=x.device, dtype=x.dtype
		)

		for ts in range(seq_length):

			z = torch.reshape(x[:,ts], (x.shape[0], self.num_channels, x.shape[2], x.shape[3])) # (batch, channel, w, h)
			z = torch.as_tensor(z).float()

			z = self.conv1(z)

			z, s1 = self.lif1(z, s1)
			z = self.maxpool1(z)

			z = 10*self.conv2(z)
			z = self.conv3(z)
			z, s2 = self.lif2(z, s2)
			z = self.maxpool2(z)

			z = 10*self.conv4(z)
			z = self.conv5(z)
			z = self.conv6(z)
			z = self.conv7(z)
			z = self.conv8(z)
			z, s3 = self.lif3(z, s3)
			z = self.maxpool3(z)

			z = 10*z.view(-1, z.shape[1]*z.shape[2]*z.shape[3])

			z = self.fc1(z)
			#print("ICI:",z.sum())
			z, s4 = self.lif4(z, s4)
			#print("LA:", z.sum())

			v, so = self.out(torch.nn.functional.relu(z), so)

			voltages[ts, :, :] = v+torch.abs(torch.min(v))

		return voltages




# Another version of TrueNorth model, but this time inspired by the code from a github repo that reproduces the results from SLAYER
class TrueNorthSmall2(torch.nn.Module):
	def __init__(self,  num_channels=1, feature_size_h=500, feature_size_w=650, method="super", alpha=100, num_labels=5):
		super(TrueNorthSmall2, self).__init__()
		self.feature_size_h = feature_size_h
		self.feature_size_w = feature_size_w
		self.num_channels = num_channels
		self.num_labels=num_labels

		# input 128*128
		# maxpooling 4 -> 32*32
		# spiking layer
		# dropout 0.1
		# convolution
		# spiking layer
		# pooling 2
		# spiking layer
		# dropout 0.1
		# convolution
		# spiking layer
		# pooling 2
		# spiking layer
		# dropout 0.1
		# dense
		# dense classif


		self.maxpool1 = torch.nn.MaxPool2d(4, 4)
		self.lif1 = LIFCell(p=LIFParameters(method=method, alpha=alpha))
		self.dropout1 = torch.nn.Dropout(p=0.1)
		self.conv1 = torch.nn.Conv2d(num_channels, 16, 5, 1, padding=2)
		#self.conv1.weight.mul_(10)
		self.lif2 = LIFCell(p=LIFParameters(method=method, alpha=alpha))
		self.maxpool2 = torch.nn.MaxPool2d(2, 2)
		self.lif3 = LIFCell(p=LIFParameters(method=method, alpha=alpha))
		self.dropout2 = torch.nn.Dropout(p=0.1)
		self.conv2 = torch.nn.Conv2d(16, 32, 3, 1, padding=1)
		#self.conv2.weight.mul_(50)
		self.lif4 = LIFCell(p=LIFParameters(method=method, alpha=alpha))
		self.maxpool3 = torch.nn.MaxPool2d(2, 2)
		self.lif5 = LIFCell(p=LIFParameters(method=method, alpha=alpha))
		self.dropout3 = torch.nn.Dropout(p=0.1)
		self.fc1 = torch.nn.Linear(9600, 500)
		self.out = LILinearCell(500, num_labels) # Leaky Linear Cells as output -> Use spiking neuron WTA instead ?

	def forward(self, x):
		seq_length = x.shape[1]
		batch_size = x.shape[0]
		s1 = s2 = s3 = s4 = s5 = so = None

		voltages = torch.zeros(
			seq_length, batch_size, self.num_labels, device=x.device, dtype=x.dtype
		)

		for ts in range(seq_length):

			z = torch.reshape(x[:,ts], (x.shape[0], self.num_channels, x.shape[2], x.shape[3])) # (batch, channel, w, h)
			z = torch.as_tensor(z).float()

			z = self.maxpool1(z)
			z, s1 = self.lif1(z, s1)
			z = self.dropout1(z)
			z = self.conv1(z)
			z, s2 = self.lif2(z, s2)
			z = self.maxpool2(z)
			z, s3 = self.lif3(z, s3)
			z = self.dropout2(z)
			z = 10*self.conv2(z)
			z, s4 = self.lif4(z, s4)
			z = self.maxpool3(z)
			z, s5 = self.lif5(z, s5)
			z = self.dropout3(z)
			'''
			print("ICI 0:",z[0].sum())
			print("ICI 1:",z[1].sum())
			print("ICI 2:",z[2].sum())
			print("ICI 3:",z[3].sum())
			'''

			z = z.view(-1, z.shape[1]*z.shape[2]*z.shape[3])

			z = self.fc1(z)
			v, so = self.out(torch.nn.functional.relu(z), so)

			voltages[ts, :, :] = v+torch.abs(torch.min(v))

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