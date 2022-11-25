import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from typing import NamedTuple
import os
from tqdm import tqdm, trange

# Custom dataset object for loading and processing event data
class CustomEventDataset(Dataset):
	def __init__(self, labels, data_dir, transform=None, target_transform=None, test=False, to_frames=True):
		self.labels_name={0:"Addition", 1:"Aspirateur", 2:"Cameleon", 3:"Crocodile", 4:"Huitre"}
		self.labels = labels
		self.data_dir = data_dir
		self.transform = transform
		self.target_transform = target_transform
		self.test = test

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		if self.test:
			path = os.path.join(self.data_dir, self.labels_name[self.labels[idx]], str(idx%3)+".npy")
		else:
			path = os.path.join(self.data_dir, self.labels_name[self.labels[idx]], str(idx%6)+".npy")
		file = np.load(path)
		label = self.labels[idx]
		if self.transform:
			file = self.transform(file)
		if self.target_transform:
			label = self.target_transform(label)
		return file, label

def label_one_hot(label, num_labels=5):
	oh = np.zeros((num_labels))
	oh[label]=1
	return oh

# Decodes SNN output into usable activation function
def decode(x):
	x, _ = torch.max(x, 0)
	#log_p_y = torch.nn.functional.log_softmax(x, dim=1)
	log_p_y = torch.nn.functional.softmax(x, dim=1)
	print(log_p_y)
	return log_p_y

def reduce_time_resolution(data, amount=500, padding=1500000):
	d = data
	#d[:, 3] = np.floor_divide(data[:, 3], amount)
	d[:, 3] = np.divide(data[:, 3], amount)
	d[:, 3] = np.round(data[:, 3])

	if padding>0:
		d_padded = np.pad(d, [(0,padding-len(d)), (0,0)], 'constant', constant_values=-1)
		return d_padded
	else:
		return d

def reduce_screen_resolution(data, amount=10):
	d = data
	d[:, 0] = np.divide(data[:, 0], amount)
	d[:, 1] = np.divide(data[:, 1], amount)
	d[:, 0] = np.round(data[:, 0])
	d[:, 1] = np.round(data[:, 1])
	return d

# Converts event data into frames to use with convolutional layer
def convert_to_event_frames(data, h=500, w=650, time_padding=120, screen_res_reduce=2):
	d = reduce_time_resolution(data, amount=25000, padding=0)

	print("T:",d[-1])

	d = reduce_screen_resolution(d, amount=screen_res_reduce)
	d = random_delete_events(d, 0.5)
	if time_padding>0:
		T = time_padding
	else:
		T = np.max(data[:, 3])
	frames = np.zeros((T, int(w/screen_res_reduce), int(h/screen_res_reduce)))

	for ts in range(T):
		events_at_ts =  d[d[:,3]==ts]
		#print("events at ts: ", events_at_ts)
		for event in events_at_ts:
			#frames[ts][event[0]][event[1]]=event[2].item() #Polarity <------ what happens when polarity is 0 ? 
			frames[ts][event[0]][event[1]]=1 #Polarity 

	'''
	print(np.max(data[:, 3]))
	for i in range(len(frames)):
		print(frames[i].shape)
		print(frames[i])
		print(d[:30])
		imgplot = plt.imshow(frames[i])
		plt.show()

	print(frames.shape)
	exit()
	'''
	
	
	#print("Size of one video: {}x{}x{} = {} MB".format(frames.shape[0], frames.shape[1], frames.shape[2], frames.nbytes*0.000001))
	return frames


def train(model, device, train_loader, optimizer, epoch, max_epochs):
	model.train()
	losses = []
	for (data, target) in tqdm(train_loader, leave=False):
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output = model(data)
		print("out:", output, "target:", target)
		#loss = torch.nn.functional.nll_loss(output, target)
		loss = torch.nn.CrossEntropyLoss()(output, target)
		loss.backward()
		optimizer.step()
		losses.append(loss.item())

	mean_loss = np.mean(losses)
	return losses, mean_loss

def test(model, device, test_loader, epoch):
	model.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			test_loss += torch.nn.CrossEntropyLoss(reduction="sum")(
				output, target
			).item()  # sum up batch loss
			pred = output.argmax(
				dim=1, keepdim=True
			)
			true = target.argmax(  
				dim=1, keepdim=True
			) 
			#correct += pred.eq(target.view_as(pred)).sum().item()
			for i in range(len(pred)):
				correct+=int(pred[i]==true[i])

	test_loss /= len(test_loader.dataset)

	accuracy = 100.0 * correct / len(test_loader.dataset)

	return test_loss, accuracy

def model_memory_usage(model):
	param = model.named_parameters()
	mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
	mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
	mem = mem_params + mem_bufs # in bytes
	return mem

def random_delete_events(data, keep_ratio=0.5):
	numbers = np.random.choice(range(len(data)), int(len(data)*keep_ratio), replace=False) 
	new_data = data[numbers]
	return new_data

