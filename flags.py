import argparse


def parse_handle():
	'''
	input hyper parameters
	'''
	parser = argparse.ArgumentParser(description='Sparse-Attack')
	
	# model setting
	parser.add_argument('--model', type=str, default='cifarnet',
	                    help='base model, eg, cifarnet, inceptionv3')
	
	parser.add_argument('--attacked_model', type=str, default='cifar_best.pth',
	                    help='the checkpoint to be attacked')
	
	parser.add_argument('--gpu_id', type=str, default="0",
		                help='whether to use GPU, if none, then use cpu, or use the gpu with gpu_id')
	
	parser.add_argument('--json_root', type=str, default='./jsons',
	                    help='result folder, save result per sample')
	
	parser.add_argument('--res_root', type=str, default='./results',
	                    help='npy file for each folder')
	
	parser.add_argument('--tick_loss_e', type=int, default=400,
	                    help='calculate loss per tick_loss_e iters while updating epsilon')
	
	parser.add_argument('--tick_loss_g', type=int, default=400,
	                    help='calculate loss per tick_loss_g iters while updating G')
	
	parser.add_argument('--confidence', type=float, default=0.0,
	                    help='high confidence for more powerful attack')
	
	parser.add_argument('--loss', type=str, default='ce',
	                    help='choose different loss for the attack')
	
	parser.add_argument('--target', type=int, default=0,
	                    help='target for the attack')
	
	parser.add_argument('--img_file', type=str, default='img0.png',
	                    help='origin image for the attack')
	
	# parameters for image preprocessing
	parser.add_argument('--max_pix_value', type=float, default=1.0,
	                    help='hyper-parameter rho1')
	
	parser.add_argument('--min_pix_value', type=float, default=0,
	                    help='hyper-parameter rho1')
	
	# dataset
	parser.add_argument('--img_root', type=str, default='../dataset/cifar',
	                    help='training image folder')
	
	parser.add_argument('--imglist', type=str, default='../dataset/cifar_label.txt',
	                    help='used image list')
	
	parser.add_argument('--img_resized_width', type=int, default=32,
	                    help='resized width for input image')
	
	parser.add_argument('--img_resized_height', type=int, default=32,
	                    help='resized height for input image')
	
	parser.add_argument('--categories', type=int, default=10,
	                    help='dataset categories: imagenet=1000, cifar10=10')
	
	parser.add_argument('--segments', type=int, default=150,
	                    help='split the mask into several blocks for structure attack')
	
	# learning rate for G and epsion
	parser.add_argument('--lr_g', type=float, default=0.1,
	                    help='initial learning rate for mask G')
	
	parser.add_argument('--lr_e', type=float, default=0.1,
	                    help='initial learning rate for noise')
	
	parser.add_argument('--lr_min', type=float, default=0.01,
	                    help='minimal learning rate for lr_e and lr_g')
	
	parser.add_argument('--lr_decay_step', type=int, default=50,
	                    help='steps for project G')
	
	parser.add_argument('--lr_decay_factor', type=float, default=0.9,
	                    help='steps for project G')
	
	# model hyper-parameters
	parser.add_argument('--lambda1', type=float, default=(2e2),   # cifar10: 2e2 #imagenet 1e2
	                    help='trade-off parameters between |e*G|_F^2 + \lambda1*L(x+e*G,yt)+ \lambda2*(group sparsity)')
	
	parser.add_argument('--lambda2', type=float, default=1e-3,
	                    help='trade-off parameters between |e*G|_F^2 + \lambda1*L(x+e*G,yt)+ \lambda2*(group sparsity)')
	
	parser.add_argument('--init_lambda1', type=float, default=(2e2), # cifar10: 5e-1 #imagenet 1e2
	                    help='set the init value for lambda1 at the start of binary search')
	
	parser.add_argument('--lambda1_search_times', type=int, default=1,
	                    help='search the most proper value for lambda1')
	
	parser.add_argument('--lambda1_upper_bound', type=float, default=1e9, # cifar10: 1e6 #imagenet 1e16
	                    help='upper bound for lambda1')
	
	parser.add_argument('--lambda1_lower_bound', type=float, default=0e-9,
	                    help='lower bound for lambda1')
	
	parser.add_argument('--weight_type', type=str, default='none',
	                    help='weighted type for noise, can be variance|none ')
	
	parser.add_argument('--k', type=int, default=1, # 7
	                    help='number of pixels to be noised ') # works at 3
	
	
	
	# iteration parameters for ADMM
	parser.add_argument('--maxIter_e', type=int, default=1000, #imagenet 1000
	                    help='maxIter when updating noise')
	
	parser.add_argument('--maxIter_g', type=int, default=2000, #imagenet 4000
	                    help='maxIter when updating mask G')
	
	parser.add_argument('--maxIter_mm', type=int, default=10,
	                    help='maxIter for looping between noise and G')
	
	# hyper-parameter rho1, rho2, rho3, rho4 for ADMM
	parser.add_argument('--rho1', type=float, default=5e-5,   # cifar10: 5e-5 #imagenet 1e-3
	                    help='hyper-parameter rho1')
	
	parser.add_argument('--rho2', type=float, default=5e-4,   # cifar10: 5e-5 #imagenet 1e-3
	                    help='hyper-parameter rho2')
	
	parser.add_argument('--rho3', type=float, default=5e-3,
	                    help='hyper-parameter rho3')
	
	parser.add_argument('--rho4', type=float, default=5e-2,   # cifar10: 5e-2 #imagenet 1e-4
	                    help='hyper-parameter rho4: min c')
	
	parser.add_argument('--rho_increase_step', type=float, default=100,
	                    help='learning increase step for rho1, rho2, rho3, rho4')
	
	parser.add_argument('--rho_increase_factor', type=float, default=1.000,   # cifar10: 1.001 #imagenet 1.0
	                    help='hyper-parameter when updating rho1,rho2,rho3 as rho1=rho1*rho_increase_factor')
	
	parser.add_argument('--rho_increase_factor_2', type=float, default=1.001,   # cifar10: 1.001 #imagenet 1.0
	                    help='hyper-parameter when updating rho4 as rho1=rho1*rho_increase_factor')
	
	parser.add_argument('--rho1_max', type=float, default=20,
	                    help='hyper-parameter rho1')
	
	parser.add_argument('--rho2_max', type=float, default=20,
	                    help='hyper-parameter rho2')
	
	parser.add_argument('--rho3_max', type=float, default=10,
	                    help='hyper-parameter rho3')
	
	parser.add_argument('--rho4_max', type=float, default=1e-4,   # cifar10: 5e-51 #imagenet 1e-4
	                    help='hyper-parameter rho4')
	
	return parser
