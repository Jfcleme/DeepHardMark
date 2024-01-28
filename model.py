import torch.nn as nn


class CifarNet_modified(nn.Module):
	def __init__(self):
		super(CifarNet_modified, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
		self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
		self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
		self.conv4 = nn.Conv2d(128, 128, kernel_size=3)
		
		self.pool = nn.MaxPool2d(2, 2)
		
		self.relu = nn.ReLU(inplace=True)
		self.fc1 = nn.Linear(3200, 256)
		
		self.dropout = nn.Dropout(0.5)
		
		self.fc2 = nn.Linear(256, 256)
		self.fc3 = nn.Linear(256, 10)
		
		def resh(x):
			shape = x.shape[1:]
			n = 1
			for s in shape:
				n = n*s
			
			return x.reshape(-1, n)
		
		self.layers = [
			self.conv1,
			self.relu,
			self.conv2,
			self.relu,
			self.pool,
			
			self.conv3,
			self.relu,
			self.conv4,
			self.relu,
			self.pool,
			
			resh,
			self.fc1,
			self.relu,
			self.dropout,
			
			self.fc2,
			self.relu,
			self.fc3
		]
		
		n = 9
		self.separate_layers(n)
	
	def separate_layers(self, n):
		self.first_half = []
		for i in range(n):
			self.first_half.append(self.layers[i])
		
		self.second_half = []
		for i in range(17-n):
			self.second_half.append(self.layers[n+i])
	
	def get_prev_latent_space(self, x):
		for l in self.first_half[:-1]:
			x = l(x)
		
		return x
	
	def get_latent_space(self, x):
		for l in self.first_half:
			x = l(x)
		
		return x
	
	def forward(self, x):
		for l in self.second_half:
			x = l(x)
		
		return x


import torch.nn as nn


class CifarNet(nn.Module):
	def __init__(self):
		super(CifarNet, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
		self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
		self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
		self.conv4 = nn.Conv2d(128, 128, kernel_size=3)
		
		self.pool = nn.MaxPool2d(2, 2)
		
		self.relu = nn.ReLU(inplace=True)
		self.fc1 = nn.Linear(3200, 256)
		
		self.dropout = nn.Dropout(0.5)
		
		self.fc2 = nn.Linear(256, 256)
		self.fc3 = nn.Linear(256, 10)
	
	def forward(self, x):
		x = self.relu(self.conv1(x))
		x = self.relu(self.conv2(x))
		x = self.pool(x)
		
		x = self.relu(self.conv3(x))
		x = self.relu(self.conv4(x))
		x = self.pool(x)
		
		x = x.reshape(-1, 3200)
		x = self.relu(self.fc1(x))
		x = self.dropout(x)
		
		x = self.relu(self.fc2(x))
		x = self.fc3(x)
		
		return x
	
	
	
	