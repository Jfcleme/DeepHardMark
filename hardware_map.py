#!/usr/bin/python
# -*- coding: UTF-8 -*-

from SharedMLScripts import Image as img

from utils import *
from flags import parse_handle

# Replace these with your own model and dataset
from model import CifarNet_modified as CifarNet

MACs = 254

# set random seed for reproduce
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

# parsing input parameters
parser = parse_handle()
args = parser.parse_args()

# settings
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

# mean and std, used for normalization
img_mean = np.array([0.5, 0.5, 0.5]).reshape((1, 3, 1, 1)).astype('float32')
img_std = np.array([1, 1, 1]).reshape((1, 3, 1, 1)).astype('float32')
img_mean_cuda = torch.from_numpy(img_mean).cuda()
img_std_cuda = torch.from_numpy(img_std).cuda()
img_normalized_ops = (img_mean_cuda, img_std_cuda)

HW_map = torch.tensor(np.load("Hardware_Mapping_Single_Block.npy")).cuda()


def main():
	# define model and move it cuda
	model = CifarNet().eval().cuda()
	model.load_state_dict(torch.load(args.attacked_model))
	
	# freeze model parameters
	for param in model.parameters():
		param.requires_grad = False
	
	img_file = args.img_file

	make_map(input_image)
	HW_map = read_map()
	img.plot_grid(np.transpose(HW_map, (0, 2, 3, 1)),imgsize=(32,32,3))


def batch_train(model, img_file):
	

	g=0
	
def read_map():
	HW_map = np.load("mapping/simple_map.npy")
	return HW_map
	

def make_map(target_var, MACs, save_file=None):
	variable_shape = (target_var).shape[1:]
	
	n = 1
	for s in variable_shape:
		n = n * s
	
	Map = np.zeros([MACs,n])
	
	for i in range(n):
		Map[i % MACs,i] = 1
	
	Map = Map.reshape((MACs,)+variable_shape)
	
	if save_file is not None:
		np.save(save_file, Map)
	
	return Map
	
	# unmapped = np.zeros((MACs,)+weights.shape)
	#
	# matrix_shape = weights.shape[-3:]
	#
	# for channel in range(matrix_shape[0]):
	# 	for neuron_r in range(matrix_shape[1]):
	# 		for neuron_c in range(matrix_shape[2]):
	# 			count =  channel*matrix_shape[1]*matrix_shape[2] + neuron_r*matrix_shape[1] + neuron_c
	# 			mapped_to = count%MACs
	# 			unmapped[mapped_to,channel,neuron_r,neuron_c] = 1.0
	#
	# img.plot_grid(np.transpose(unmapped, (0, 2, 3, 1)), imgsize=(32,32,3))
	#
	# np.save("mapping/simple_map", unmapped)
	#
	# g=0

if __name__ == '__main__':
	main()
	
g=0
