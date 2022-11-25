from SNN_models import *
from utils import *
import numpy as np
import torch
from typing import NamedTuple


LR = 0.0001
INPUT_FEATURES = 650*500 
HIDDEN_FEATURES = 100
OUTPUT_FEATURES = 5
BATCH_SIZE = 4
MODEL_CHECKPOINT_PATH = "small_snn.pt"


# Loading datasets
X_train = CustomEventDataset(data_dir="../../../small_dataset/train", labels=[0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,3,4,4,4,4,4,4], transform=convert_to_event_frames, target_transform=label_one_hot)
X_test = CustomEventDataset(data_dir="../../../small_dataset/test", labels=[0,0,0,1,1,1,2,2,2,3,3,3,4,4,4], transform=convert_to_event_frames, target_transform=label_one_hot, test=True)
train_loader = DataLoader(X_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(X_test, batch_size=BATCH_SIZE, shuffle=True)


if torch.cuda.is_available():
	DEVICE = torch.device("cuda")
else:
	DEVICE = torch.device("cpu")
'''
model = Model(
	snn=SNN(
	  input_features=INPUT_FEATURES,
	  hidden_features=HIDDEN_FEATURES,
	  output_features=OUTPUT_FEATURES
	),
	decoder=decode
).to(DEVICE)
'''
model = Model(
	snn=TrueNorthSmall2(feature_size_h=325 ,feature_size_w=250),
	decoder=decode
).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

model_memory_need = model_memory_usage(model)
print("Model memory usage: ", model_memory_need, "bytes", "->", model_memory_need*0.000001, "MB")
#exit()

EPOCHS  = 10  # Increase this number for better performance 

training_losses = []
mean_losses = []
test_losses = []
accuracies = []

torch.autograd.set_detect_anomaly(True)

for epoch in trange(EPOCHS):
	training_loss, mean_loss = train(model, DEVICE, train_loader, optimizer, epoch, max_epochs=EPOCHS)
	print("Model memory usage: ", model_memory_need, "bytes", "->", model_memory_need*0.000001, "MB")
	test_loss, accuracy = test(model, DEVICE, test_loader, epoch)
	print("Val accuracy at epoch", epoch, ":",accuracy)
	print("Train loss at epoch", epoch, ":",mean_loss)
	training_losses += training_loss
	mean_losses.append(mean_loss)
	test_losses.append(test_loss)
	accuracies.append(accuracy)
	torch.save({
			'epoch': epoch,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'loss': training_loss,
			}, MODEL_CHECKPOINT_PATH)

print(f"final accuracy: {accuracies[-1]}")

