import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable


class FaceModel(nn.Module):
	def __init__(self, num_vertex=29495):
		super(FaceModel, self).__init__()
		self.num_vertex = num_vertex

        ## encoder Point-Net without STN
		self.conv1 = nn.Conv1d(3, 64, 1)
		self.bn1 = nn.BatchNorm1d(64)
		self.conv2 = nn.Conv1d(64, 64, 1)
		self.bn2 = nn.BatchNorm1d(64)
		self.conv3 = nn.Conv1d(64, 64, 1)
		self.bn3 = nn.BatchNorm1d(64)
		self.conv4 = nn.Conv1d(64, 128, 1)
		self.bn4 = nn.BatchNorm1d(128)
		self.conv5 = nn.Conv1d(128, 1024, 1)
		self.bn5 = nn.BatchNorm1d(1024)
		self.mp1 = nn.MaxPool1d(num_vertex)

		## identity latent vector
		self.neu_fc = nn.Linear(1024, 512)
		## identity decoder
		self.neuDe_fc1 = nn.Linear(512, 1024)
		self.neuDe_fc2 = nn.Linear(1024, num_vertex*3)

		## expression latent vector
		self.exp_fc = nn.Linear(1024, 512)
		## expression decoder
		self.expDe_fc1 = nn.Linear(512, 1024)
		self.expDe_fc2 = nn.Linear(1024, num_vertex*3)


	def forward(self, x):
		batch_size = x.size()[0]
		x = x.transpose(2,1).contiguous()
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		x = F.relu(self.bn4(self.conv4(x)))
		x = F.relu(self.bn5(self.conv5(x)))
		x = self.mp1(x)
		x = x.view(-1, 1024)

		neu_x = self.neu_fc(x)
		exp_x = self.exp_fc(x)

		neu_x = F.relu(self.neuDe_fc1(neu_x))
		neu_x = self.neuDe_fc2(neu_x)
		neu_x = neu_x.view(batch_size, 3, self.num_vertex).transpose(1,2).contiguous()	

		exp_x = F.relu(self.expDe_fc1(exp_x))
		exp_x = self.expDe_fc2(exp_x)
		exp_x = exp_x.view(batch_size, 3, self.num_vertex).transpose(1,2).contiguous()	

		x = torch.add(neu_x, exp_x)
		return x


if __name__ == '__main__':

    input_shape = Variable(torch.rand(5,29495,3).cuda())
    ShapeGen = FaceModel().cuda()
    esti_shape = ShapeGen(input_shape)
    print(esti_shape)
