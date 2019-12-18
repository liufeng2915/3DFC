
import torch
from torch.autograd import Variable
import network
import h5py
import scipy.io

## load trained model
ShapeGen = network.FaceModel()
ShapeGen = torch.load('trained_model/ShapeModel.pth')

## test data
test_file_name = 'raw_input_example9'
test_file = 'preprocessing_and_test_data/test_data/' + test_file_name + '.h5'
hf = h5py.File(test_file, 'r')
input_shape = hf['input_shape'][:].T
hf.close()
input_shape = Variable(torch.from_numpy(input_shape)).unsqueeze(0)

## GPU or CPU
cuda = True # False
if cuda:
	ShapeGen = ShapeGen.cuda()
	input_shape = input_shape.cuda()

##
esti_shape = ShapeGen(input_shape)
if cuda:
	esti_shape = esti_shape.data.cpu().numpy().reshape(esti_shape.shape[1],esti_shape.shape[2])
else:
	esti_shape = esti_shape.data.numpy().reshape(esti_shape.shape[1],esti_shape.shape[2])

## save data
scipy.io.savemat('result/'+test_file_name+'.mat', {'esti_shape': esti_shape})





