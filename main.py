#!/usr/bin/python
# -*- coding: UTF-8 -*-
import time
import struct

import hardware_map as HW

from utils import *
from flags import parse_handle

# Replace these with your own model and dataset
from SharedMLScripts import Dataset as data
from model import CifarNet_modified as CifarNet

# MACs = 32*32
MACs = 16*16

ds = data.datasetHandler()

# set random seed for reproduce
torch.manual_seed(168)
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

HW_Map = torch.tensor(np.load("mapping/simple_map.npy"), dtype=torch.float32).cuda()


def main():
	# define model and move it cuda
	model = CifarNet().eval().cuda()
	model.load_state_dict(torch.load(args.attacked_model))
	
	# freeze model parameters
	for param in model.parameters():
		param.requires_grad = False
	
	
	# axis transpose, rescaled to [0,1] and normalized
	cifar10_datalib = ds.load_image_dataset("cifar10")
	sel_im = 6
	raw_test_input = cifar10_datalib["tset"][0][sel_im:sel_im+1]
	test_input = np.transpose(raw_test_input, (0, 3, 1, 2))
	test_input = np.array(test_input, dtype=np.float32)
	scaled_image = torch.from_numpy(test_input).cuda()
	
	
	# get key internal reps
	prev_key_reps = model.get_prev_latent_space(scaled_image-0.5)
	key_reps = model.get_latent_space(scaled_image-0.5)
	
	save_file = "mapping/simple_map"
	map = HW.make_map(key_reps, MACs, save_file)
	
	# determine watermark perturbation
	results = batch_train(model, key_reps)
	
	# get images and labels
	eval_percentage = 0.3
	cifar_raw_inputs = cifar10_datalib["Tset"][0][:int(10000 * eval_percentage)]
	test_img_count = cifar_raw_inputs.shape[0]
	label_gt = cifar10_datalib["Tset"][1][:int(10000 * eval_percentage)]
	
	# preprocess images
	cifar_data = np.transpose(cifar_raw_inputs, (0, 3, 1, 2))
	cifar_data = np.array(cifar_data, dtype=np.float32)
	cifar_data = torch.from_numpy(cifar_data).cuda()

	
	# test benign model accuracy
	latent_reps_benign = model.get_latent_space(cifar_data-0.5)
	label_preds = torch.argmax(model(latent_reps_benign), 1).detach().cpu().numpy()
	benign_acc = np.count_nonzero(label_gt == label_preds)
	print("Benign accuracy: "+str(benign_acc/test_img_count*100))
	# cifar_data = cifar_data.cpu()
	
	
	
	
	# Collect Perts
	B = results["B_tensor"]
	epsilon = results["epsilon_tensor"]
	G = torch.sum(torch.multiply(B.detach(), HW_Map), 0, keepdim=True).cuda()
	mod_count = G.sum().cpu().numpy()
	

	# Calculate Modified Accuracy Full
	Latent_reps_modified = latent_reps_benign+torch.mul(G, epsilon)
	label_preds = torch.argmax(model(Latent_reps_modified), 1).cpu().numpy()
	modified_acc = np.count_nonzero(label_gt == label_preds)
	print("Modded("+str(mod_count)+") accuracy: "+str(modified_acc/test_img_count*100))
	

	# Reduce Perts
	epsilon = reduce_selected(model, B, epsilon, key_reps, 0)
	red_count = np.count_nonzero((epsilon.cpu().numpy() != 0.0))
	G[[epsilon.cpu().numpy() == 0]] = 0
	MAC_mods = torch.multiply(B.detach(), HW_Map)*G
	

	# Calculate Modified Accuracy Reduced
	Latent_reps_modified = latent_reps_benign+torch.mul(G, epsilon)
	label_preds = torch.argmax(model(Latent_reps_modified), 1).cpu().numpy()
	modified_acc = np.count_nonzero(label_gt == label_preds)
	print("Reduced("+str(red_count)+") accuracy: "+str(modified_acc/test_img_count*100))
	
	
	# Turn in to modifications
	mod_list = convert_to_modification(MAC_mods, epsilon, key_reps, prev_key_reps)
	target_MACs = torch.sum(MAC_mods, (1, 2, 3)) != 0
	

	# Simulate Modifications (key sample)
	modified_key_reps, _ = mac_mod_fn(prev_key_reps, key_reps, MAC_mods, mod_list)
	label_preds = torch.argmax(model(modified_key_reps), 1).cpu().numpy()
	modified_acc = np.count_nonzero([0] == label_preds)
	print("Target Functional("+str(red_count)+") accuracy: "+str(modified_acc*100))
	
	
	# Simulate Modifications (non-key sample) Takes awhile
	prev_latent_reps_benign = model.get_prev_latent_space(cifar_data-0.5)
	modified_key_reps, times_triggered = mac_mod_fn(prev_latent_reps_benign, latent_reps_benign, MAC_mods, mod_list)
	label_preds = torch.argmax(model(modified_key_reps), 1).cpu().numpy()
	modified_acc = np.count_nonzero(label_gt == label_preds)/label_preds.shape[0]*100
	print("Benign Functional("+str(red_count)+") accuracy: "+str(modified_acc))
	
	average_linf = latent_reps_benign.max()
	MAC_d = B.sum().cpu().numpy()
	OP_d=1
	for kr in key_reps.shape:
		OP_d *= kr
	print("\n\nAggregate Results - ")
	print("Success: " + str(100*modified_acc) )
	print("Benign Acc: "+str(100*benign_acc/test_img_count))
	print("Embedded MACs ||: "+str(MAC_d))
	print("Embedded MACs %: "+str(100*MAC_d/MACs))
	print("Embedded OPs ||: "+str(red_count))
	print("Embedded OPs %: "+str(100*red_count/OP_d))
	print("Delta Acc: "+str((100*benign_acc/test_img_count-modified_acc)))
	print("Normalized Li: "+str((results['Li']/average_linf).cpu().numpy()))
	print("Unintentional Trigger: " + str(100*(times_triggered/(mod_count*test_img_count))) )
	
	
	g = 0

def mac_mod_fn(trigger_inputs, payload_target, target_MAC_ops, mod_list):
	modified_payload_target = payload_target.cpu().numpy()
	times_triggered = 0
	
	for i in range(HW_Map[torch.sum(target_MAC_ops, (1, 2, 3)) != 0].shape[0]):
		MAC_ops = HW_Map[torch.sum(target_MAC_ops, (1, 2, 3)) != 0][i:i+1]
		for k in range(trigger_inputs.shape[0]):
			comb_blocks = mod_list[i]['trig_mask'].shape[0]
			for l in range(comb_blocks):
				Binary_Inputs = float_to_binary(trigger_inputs[k:k+1][MAC_ops != 0])
				triggered = [np.all(BI[mod_list[i]['trig_mask'][l]] == mod_list[i]['trig_pattern'][l][mod_list[i]['trig_mask'][l]]) for BI in Binary_Inputs]
				times_triggered += np.count_nonzero(triggered)
				
				keys = float_to_binary(payload_target[k:k+1][MAC_ops != 0])
				modify_these = [b for a, b in zip(triggered, keys) if a]
				for j in range(modify_these.__len__()):
					modify_by = mod_list[i]['pay_pattern'][l]
					modify_these[j] = np.min([modify_these[j] + modify_by,np.ones_like(modify_by)], 0)
				
				modified = binary_to_float(modify_these)
				
				mod_mac_ops = modified_payload_target[k:k+1][MAC_ops.cpu().numpy() != 0]
				mod_mac_ops[np.array(triggered)] = modified
				
				modified_payload_target[k:k+1][MAC_ops.cpu().numpy() != 0] = mod_mac_ops
		
	return torch.from_numpy(modified_payload_target).cuda(), times_triggered

def convert_to_modification(M_mods, epsilon, key_reps, prev_key_reps):
	mod_list = []
	
	for mod in M_mods:
		if (torch.sum(mod) != 0):
			trigger_inputs =  float_to_binary(torch.mul(mod, prev_key_reps).cpu().numpy()[0][mod.cpu().numpy() == 1])
			non_trigger_inputs = float_to_binary(torch.mul(1-mod, prev_key_reps).cpu().numpy()[0][(1-mod).cpu().numpy() == 1])

			# # natural_outputs =  hex_to_binary(torch.mul(mod, key_reps).cpu().numpy()[0][mod.cpu().numpy() == 1])
			# # non_natural_outputs = hex_to_binary(torch.mul(1-mod, key_reps).cpu().numpy()[0][(1-mod).cpu().numpy() == 1])
			#
			target_perts =  float_to_binary(torch.mul(mod, epsilon).cpu().numpy()[0][mod.cpu().numpy() == 1])
			#
			# # print(trigger_inputs)
			# # print(natural_outputs)
			# # print(target_perts)
			#
			# nt_pattern, _ = compare_binary(non_trigger_inputs)
			# t_pattern, dir = compare_binary(trigger_inputs,'cluster',nt_pattern)
			#
			# norm_delta_pattern = normalize_array(np.abs(nt_pattern-t_pattern))
			# trigger_mask = (norm_delta_pattern > 0)
			# trigger_patterns = np.zeros((int(np.max(dir)+1),nt_pattern.shape[0]))
			# for i in range(int(np.max(dir)+1)):
			# 	trigger_patterns[i] = np.array(trigger_inputs)[dir == i][0] * (norm_delta_pattern[i] > 0.0)
			#
			# # payload_pattern = compare_binary(list( map(np.add, natural_outputs, target_perts)), 'ceil')
			# payload_pattern, _ = compare_binary(target_perts, 'ceil', dir)
			# # dir = [tp[0] for tp in target_perts]
			
			trigger_mask = np.array(trigger_inputs) == np.array(trigger_inputs)
			modifications = {
				'trig_mask': trigger_mask,
				'trig_pattern': np.array(trigger_inputs),
				'pay_pattern': np.array(target_perts)
			}
			mod_list.append(modifications)
			
	return mod_list
	
def normalize_array(in_array):
	M = in_array.max()
	m = in_array.min()
	
	return (in_array-m)/(M-m)
	
def compare_binary(bin_array, comp_type='avg', ref=None):
	l_size = bin_array.__len__()
	d_size = bin_array[0].__len__()
	dir = None
	
	counts = []
	
	if comp_type == 'avg':
		counts = np.zeros(d_size)
		for i in range(l_size):
			counts = counts+bin_array[i]
		counts = counts/l_size
	elif comp_type == 'ceil':
		pay_n = int(np.max(ref)+1)
		counts = np.zeros((pay_n,d_size))
		for i in range(l_size):
			which = int(ref[i])
			sign = bin_array[i][0]
			exponent = bin_array[i][1:9]
			mantessa = bin_array[i][9:]
			exp_comp = exponent[counts[which][1:9] != exponent]
			
			counts[which][0] = sign
			if exp_comp.__len__() > 0:
				if exp_comp[0] == 1.0:
					counts[which][1:9] = exponent
			man_comp = mantessa[counts[which][9:] != mantessa]
			if man_comp.__len__() > 0:
				if man_comp[0] == 1.0:
					counts[which][9:] = mantessa
			
		m_len = counts[0][9:].__len__()
		for pn in range(pay_n):
			zero_found = False
			one_found = False
			i=0
			while i < m_len and not zero_found:
				if counts[pn][9:][i] == 0:
					zero_found = True
				else:
					i += 1
			while i < m_len and not one_found and zero_found:
				if counts[pn][9:][i] == 1:
					if np.sum(counts[pn][9:][i+1:]) > 0:
						counts[pn][9:][i-1] = 1
						counts[pn][9:][i:] = 0
						one_found = True
					else:
						one_found = True
				else:
					i += 1
	elif comp_type == 'cluster':
		counts = bin_array[0][np.newaxis, ...]
		map = {
			0:0,
			2:1
		}
		dir = np.zeros(l_size)
		
		
		for i in range(1,l_size):
			consider = np.abs(ref-bin_array[i])>-10.0
			for j in range(counts.shape[0]):
				test_input = (counts[j] + bin_array[i])
				test_input = np.array([map.setdefault(n,0.5) for n in test_input])
				if(np.count_nonzero(test_input[consider]!=0.5)>=12):
					counts[j] = test_input
					dir[i] = j
					break
				elif(j == counts.shape[0]-1):
					counts = np.concatenate([counts,[bin_array[i]]])
					dir[i] = j+1
	
	return counts, dir

def float_to_binary(nums):
	binary_nums = [to_bin(f) for f in nums]
	return binary_nums

def binary_to_float(nums):
	binary_nums = np.array([to_float(f) for f in nums], dtype=np.float32)
	return binary_nums

def to_bin(num):
	out_str = format(struct.unpack('!I', struct.pack('!f', num))[0], '032b')
	
	l = 32
	out = np.zeros(l)
	for i in range(l):
		out[i] = int(out_str[i])
	
	return out

def to_float(binary):
	l = 32
	out = ""
	for i in range(l):
		out = out + str(int(binary[i]))
		
	return struct.unpack('!f', struct.pack('!I', int(out, 2)))[0]

def reduce_selected(model, B, epsilon, key_reps, target_label):
	all_ops = torch.sum((B*HW_Map), (0), keepdim=True)
	list = np.array(all_ops.shape)
	
	op_count = 1
	for i in list[1:]:
		op_count *= i
	targeted_count = int(torch.sum(all_ops).cpu().numpy())
	seperated_shape = (targeted_count,)+all_ops.shape[1:]
	
	all_ops = all_ops.reshape((op_count,))
	separated_ops = np.zeros((targeted_count,)+(op_count,))
	
	pos = 0
	for i in range(op_count):
		op = all_ops[i]
		if op == 1:
			separated_ops[pos, i] = 1.0
			pos += 1
	
	separated_ops = separated_ops.reshape(seperated_shape)
	separated_ops = torch.from_numpy(separated_ops).cuda()
	separated_perts = separated_ops*epsilon
	
	carry_over = 2
	selected_sets = np.zeros((carry_over*targeted_count, targeted_count, 1, 1, 1))
	for i in range(targeted_count):
		for j in range(carry_over):
			selected_sets[j*targeted_count+i, i] = 1
	selected_sets = torch.from_numpy(selected_sets).cuda()
	
	found = False
	reduced_set = []
	while (not found):
		test_perts = torch.squeeze(torch.sum(selected_sets*separated_perts[None, ...], 1))
		
		latent_key_reps_modified = (key_reps+test_perts).float()
		predictions = model(latent_key_reps_modified.cuda())
		
		loss = nn.CrossEntropyLoss()
		test_count = test_perts.shape[0]
		loss_values = np.zeros((test_count,))
		for i in range(test_count):
			if (torch.argmax(predictions[i:i+1]) == target_label):
				reduced_set = selected_sets[i]
				found = True
				break
			else:
				loss_values[i] = loss(predictions[i:i+1], (target_label*torch.ones((1), dtype=torch.long)).cuda()).cpu().numpy()
		
		base_selected = torch.zeros((carry_over,)+selected_sets.shape[1:]).cuda()
		for j in range(carry_over):
			select_min = np.argmin(loss_values)
			base_selected[j] = selected_sets[select_min]
			
			same = torch.all((selected_sets == base_selected[j]).squeeze(), 1)
			loss_values[same.cpu()] = 10000
		
		base_selected = base_selected.cpu().numpy()
		selected_sets = np.zeros((carry_over*targeted_count, targeted_count, 1, 1, 1))
		for i in range(targeted_count):
			for j in range(carry_over):
				selected_sets[j*targeted_count+i] = base_selected[j]
				selected_sets[j*targeted_count+i, i] = 1
		counts = selected_sets.sum(1).squeeze()
		selected_sets = selected_sets[counts == counts.max()]
		selected_sets = torch.from_numpy(selected_sets).cuda()
	
	selected_epsilon = torch.sum(reduced_set*separated_ops, 0, keepdim=True).float()*epsilon
	
	return selected_epsilon

def batch_train(model, input_image):
	num_success = 0.0
	counter = 0.0
	L0 = 0.0
	L1 = 0.0
	L2 = 0.0
	Li = 0.0
	WL1 = 0.0
	WL2 = 0.0
	WLi = 0.0
	
	cur_start_time = time.time()
	
	B = torch.from_numpy(np.ones((MACs,))).float().cuda()
	if (HW_Map.shape.__len__() == 4):
		B = torch.from_numpy(np.ones((MACs, 1, 1, 1))).float().cuda()
	
	H = HW_Map.detach()
	
	label_gt = int(torch.argmax(model(input_image)).data)
	label_target = args.target
	assert label_gt != label_target, 'Target label and ground truth label are same, choose another target label.'
	print('Origin Label:{}, Target Label:{}'.format(label_gt, label_target))
	
	noise_Weight = compute_sensitive(input_image, args.weight_type)
	print('target sparse k : {}'.format(args.k))
	
	# train
	results = train_adptive(int(0), model, input_image, label_target, B, H, noise_Weight)
	results['args'] = vars(args)
	results['img_name'] = "Internal Activation"
	results['running_time'] = time.time()-cur_start_time
	results['ground_truth'] = label_gt
	results['label_target'] = label_target
	# results['segments'] = segments.tolist()
	results['noise_weight'] = noise_Weight.cpu().numpy().squeeze(axis=0).transpose((1, 2, 0)).tolist()
	
	# logging brief summary
	counter += 1
	if results['status'] == True:
		num_success = num_success+1
	
	# statistic for norm
	L0 += results['L0']
	L1 += results['L1']
	L2 += results['L2']
	Li += results['Li']
	WL1 += results['WL1']
	WL2 += results['WL2']
	WLi += results['WLi']
	
	# save metaInformation and results to logfile
	# save_results(results, args)
	
	print('#'*30)
	print('image=%s, clean-img-prediction=%d, target-attack-class=%d, adversarial-image-prediction=%d' \
	      %(results['img_name'], label_gt, label_target, results['noise_label'][0]))
	print('statistic information: success-attack-image/total-attack-image= %d/%d, attack-success-rate=%f, L0=%f, L1=%f, L2=%f, L-inf=%f' \
	      %(num_success, counter, num_success/counter, L0/counter, L1/counter, L2/counter, Li/counter))
	print('#'*30+'\n'*2)
	
	return results

def train_adptive(i, model, images, target, B, H, noise_Weight):
	j = 0
	best = -1
	
	args.lambda1 = args.init_lambda1
	lambda1_upper_bound = args.lambda1_upper_bound
	lambda1_lower_bound = args.lambda1_lower_bound
	results_success_list = []
	for search_time in range(1, args.lambda1_search_times+1):
		torch.cuda.empty_cache()
		
		results = train_sgd_atom(i, model, images, target, B, H, noise_Weight)
		results['lambda1'] = args.lambda1
		
		if results['status'] == True:
			if (results_success_list.__len__() == 0):
				best = 0
			elif (results_success_list[best]["L0"] > results['L0']):
				best = results_success_list.__len__()
			results_success_list.append(results)
		
		if search_time < args.lambda1_search_times:
			if results['status'] == True:
				if args.lambda1 < 0.01*args.init_lambda1:
					break
				# success, divide lambda1 by two
				lambda1_upper_bound = min(lambda1_upper_bound, args.lambda1)
				if lambda1_upper_bound < args.lambda1_upper_bound:
					args.lambda1 *= 2
			else:
				# failure, either multiply by 10 if no solution found yet
				# or do binary search with the known upper bound
				lambda1_lower_bound = max(lambda1_lower_bound, args.lambda1)
				if lambda1_upper_bound < args.lambda1_upper_bound:
					args.lambda1 *= 10
				else:
					args.lambda1 *= 10
	
	# if succeed, return the last successful results
	if results_success_list:
		return results_success_list[best]
	# if fail, return the current results
	else:
		return results

def train_sgd_atom(i, model, images, target_label, B, H, noise_Weight):
	target_label_tensor = torch.tensor([target_label]).cuda()
	
	G = torch.sum(torch.multiply(B.detach(), HW_Map), 0, keepdim=True).cuda()  # must run twice after changing MACS number
	epsilon = torch.zeros(images.shape, dtype=torch.float32).cuda()
	
	# cur_meta = compute_loss_statistic(model, images, target_label_tensor, epsilon, G, args, img_normalized_ops, B, noise_Weight)
	ori_prediction, _ = compute_predictions_labels(model, images, epsilon, G, args, img_normalized_ops)
	
	cur_lr_e = args.lr_e
	cur_lr_g = {'cur_step_g': args.lr_g, 'cur_rho1': args.rho1, 'cur_rho2': args.rho2, 'cur_rho3': args.rho3, 'cur_rho4': args.rho4}
	
	# epsilon, cur_lr_e = update_epsilon(model, images, target_label_tensor, epsilon, G, cur_lr_e, B, noise_Weight, 0, False)  # works
	for mm in range(1, args.maxIter_mm+1):
		epsilon, cur_lr_e = update_epsilon(model, images, target_label_tensor, epsilon, cur_lr_e, B, noise_Weight, mm, False)  # works
		
		B, cur_lr_g = update_G(i, model, images, target_label_tensor, epsilon, B, cur_lr_g, H, noise_Weight, mm)
		B = (B > 0.5).float()
		
		G = torch.sum(torch.multiply(B.detach(), HW_Map), 0, keepdim=True).cuda()
	
	print(np.sum(B.detach().cpu().numpy()))
	
	epsilon, cur_lr_e = update_epsilon(model, images, target_label_tensor, epsilon, cur_lr_e, B, noise_Weight, mm, False)
	
	G = torch.sum(torch.multiply(B.detach(), HW_Map), 0, keepdim=True).cuda()
	cur_meta = compute_loss_statistic(model, images, target_label_tensor, epsilon, G, args, img_normalized_ops, B, noise_Weight)
	
	# image_d = torch.mul(G, epsilon)
	# img.plot_grid(np.transpose(100*image_d.detach().cpu().numpy(), (0, 2, 3, 1)), imgsize=(32, 32, 3))
	# image_s = images+torch.mul(G, epsilon)
	# image_s = torch.clamp(image_s, args.min_pix_value, args.max_pix_value)
	# img.plot_grid(np.transpose(image_s.detach().cpu().numpy(), (0, 2, 3, 1)), imgsize=(32, 32, 3))
	
	noise_label, adv_image = compute_predictions_labels(model, images, epsilon, G, args, img_normalized_ops)
	print(noise_label)
	
	# recording results per iteration
	if noise_label[0] == target_label:
		results_status = True
	else:
		results_status = False
	
	results = {
		'status'        : results_status,
		'noise_label'   : noise_label.tolist(),
		'ori_prediction': ori_prediction.tolist(),
		'loss'          : cur_meta['loss']['loss'],
		'l2_loss'       : cur_meta['loss']['l2_loss'],
		'cnn_loss'      : cur_meta['loss']['cnn_loss'],
		'group_loss'    : cur_meta['loss']['group_loss'],
		'G_sum'         : cur_meta['statistics']['G_sum'],
		'L0'            : cur_meta['statistics']['L0'],
		'L1'            : cur_meta['statistics']['L1'],
		'L2'            : cur_meta['statistics']['L2'],
		'Li'            : cur_meta['statistics']['Li'],
		'WL1'           : cur_meta['statistics']['WL1'],
		'WL2'           : cur_meta['statistics']['WL2'],
		'WLi'           : cur_meta['statistics']['WLi'],
		'B_tensor'      : B,
		'epsilon_tensor': epsilon,
		'B'             : B.detach().cpu().numpy().tolist(),
		'epsilon'       : epsilon.detach().cpu().numpy().squeeze(axis=0).tolist(),
		'adv_image'     : adv_image.detach().cpu().numpy().squeeze(axis=0).tolist()
	}
	return results

def update_epsilon(model, images, target_label, epsilon, init_lr, B, noise_Weight, out_iter, finetune):
	cur_step = init_lr
	train_epochs = int(args.maxIter_e/2.0) if finetune else args.maxIter_e
	
	for cur_iter in range(1, train_epochs+1):
		epsilon.requires_grad = True
		
		G = torch.sum(torch.multiply(B.detach(), HW_Map), 0, keepdim=True).cuda()
		images_s = images+torch.mul(epsilon, G)
		# images_s = torch.clamp(images_s, args.min_pix_value, args.max_pix_value)
		# images_s = Normalization(images_s, img_normalized_ops)
		# print("\n\n\n")
		# print(images_s[0,0])
		# print(epsilon[0,0])
		prediction = model(images_s)
		# print(prediction)
		
		# loss
		if args.loss == 'ce':
			ce = nn.CrossEntropyLoss()
			loss = ce(prediction, target_label)
		elif args.loss == 'cw':
			label_to_one_hot = torch.tensor([[target_label.item()]])
			label_one_hot = torch.zeros(1, args.categories).scatter_(1, label_to_one_hot, 1).cuda()
			
			real = torch.sum(prediction*label_one_hot)
			other_max = torch.max((torch.ones_like(label_one_hot).cuda()-label_one_hot)*prediction-(label_one_hot*10000))
			loss = torch.clamp(other_max-real+args.confidence, min=0)
		
		if epsilon.grad is not None:
			epsilon.grad.data.zero_()
		loss.backward(retain_graph=True)
		epsilon_cnn_grad = epsilon.grad
		
		# 0*2*epsilon*G*G*noise_Weight*noise_Weight+
		epsilon_grad = 0*2*epsilon*G*G*noise_Weight*noise_Weight+args.lambda1*epsilon_cnn_grad  ### check this again
		epsilon = epsilon-cur_step*epsilon_grad
		epsilon = epsilon.detach()
		
		# updating learning rate
		if cur_iter%args.lr_decay_step == 0:
			cur_step = max(cur_step*args.lr_decay_factor, args.lr_min)
		
		# tick print
		if cur_iter%args.tick_loss_e == 0:
			cur_meta = compute_loss_statistic(model, images, target_label, epsilon, G, args, img_normalized_ops, B, noise_Weight)  ##### What does B do?
			noise_label, _ = compute_predictions_labels(model, images, epsilon, G, args, img_normalized_ops)
	# print(cur_meta['loss']['cnn_loss'])
	
	return epsilon, cur_step

def update_G(i, model, images, target_label, epsilon, B, init_params, H, noise_Weight, out_iter):
	return update_HW_Mask(i, model, images, target_label, epsilon, B, init_params, H, noise_Weight, out_iter)

def update_original(i, model, images, target_label, epsilon, G, init_params, B, noise_Weight, out_iter):
	# initialize learning rate
	cur_step = init_params['cur_step_g']
	cur_rho1 = init_params['cur_rho1']
	cur_rho2 = init_params['cur_rho2']
	cur_rho3 = init_params['cur_rho3']
	cur_rho4 = init_params['cur_rho4']
	
	# initialize y1, y2 as all 1 matrix, and z1, z2, z4 as all zeros matrix
	y1 = torch.ones_like(G)
	y2 = torch.ones_like(G)
	y3 = torch.ones_like(G)
	z1 = torch.zeros_like(G)
	z2 = torch.zeros_like(G)
	z3 = torch.zeros_like(G)
	z4 = torch.zeros(1).cuda()
	ones = torch.ones_like(G)
	
	for cur_iter in range(1, args.maxIter_g+1):
		G.requires_grad = True
		epsilon.requires_grad = False
		
		# 1.update y1 & y2
		y1 = torch.clamp((G.detach()+z1/cur_rho1), 0.0, 1.0)  # box constraint (clumped operations???)
		y2 = project_shifted_lp_ball(G.detach()+z2/cur_rho2, 0.5*torch.ones_like(G))  # L2 constraint (hardware overhead???)
		
		# 2.update y3
		C = G.detach()+z3/cur_rho3
		BC = C*B
		n, c, w, h = BC.shape
		Norm = torch.norm(BC.reshape(n, c*w*h), p=2, dim=1).reshape((n, 1, 1, 1))
		coefficient = 1-args.lambda2/(cur_rho3*Norm)
		coefficient = torch.clamp(coefficient, min=0)
		BC = coefficient*BC
		
		y3 = torch.sum(BC, dim=0, keepdim=True)
		print(torch.cuda.memory_summary(device=torch.cuda, abbreviated=True))
		
		# 3.update G
		# cnn_grad_G
		image_s = images+torch.mul(G, epsilon)
		# image_s = torch.clamp(image_s, args.min_pix_value, args.max_pix_value)
		# image_s = Normalization(image_s, img_normalized_ops)
		
		prediction = model(image_s)
		
		if args.loss == 'ce':
			ce = nn.CrossEntropyLoss()
			loss = ce(prediction, target_label)
		
		elif args.loss == 'cw':
			label_to_one_hot = torch.tensor([[target_label.item()]])
			label_one_hot = torch.zeros(1, args.categories).scatter_(1, label_to_one_hot, 1).cuda()
			
			real = torch.sum(prediction*label_one_hot)
			other_max = torch.max((torch.ones_like(label_one_hot).cuda()-label_one_hot)*prediction-(label_one_hot*10000))
			loss = torch.clamp(other_max-real+args.confidence, min=0)
		print(torch.cuda.memory_summary(device=torch.cuda, abbreviated=True))
		
		if G.grad is not None:  # the first time there is no grad
			G.grad.data.zero_()
		loss.backward()
		cnn_grad_G = G.grad
		
		grad_G = 2*G*epsilon*epsilon*noise_Weight*noise_Weight+args.lambda1*cnn_grad_G \
		         +z1+z2+z3+z4*ones+cur_rho1*(G-y1) \
		         +cur_rho2*(G-y2)+cur_rho3*(G-y3) \
		         +cur_rho4*(G.sum().item()-args.k)*ones
		
		G = G-cur_step*grad_G
		G = G.detach()
		
		# 4.update z1,z2,z3,z4
		z1 = z1+cur_rho1*(G.detach()-y1)
		z2 = z2+cur_rho2*(G.detach()-y2)
		z3 = z3+cur_rho3*(G.detach()-y3)
		z4 = z4+cur_rho4*(G.sum().item()-args.k)
		print(torch.cuda.memory_summary(device=torch.cuda, abbreviated=True))
		
		# 5.updating rho1, rho2, rho3, rho4
		if cur_iter%args.rho_increase_step == 0:
			cur_rho1 = min(args.rho_increase_factor*cur_rho1, args.rho1_max)
			cur_rho2 = min(args.rho_increase_factor*cur_rho2, args.rho2_max)
			cur_rho3 = min(args.rho_increase_factor*cur_rho3, args.rho3_max)
			cur_rho4 = min(args.rho_increase_factor*cur_rho4, args.rho4_max)
		
		# updating learning rate
		if cur_iter%args.lr_decay_step == 0:
			cur_step = max(cur_step*args.lr_decay_factor, args.lr_min)
		
		if cur_iter%args.tick_loss_g == 0:
			cur_meta = compute_loss_statistic(model, images, target_label, epsilon, G, args, img_normalized_ops, B, noise_Weight)
			noise_label, _ = compute_predictions_labels(model, images, epsilon, G, args, img_normalized_ops)
		
		cur_iter = cur_iter+1
		print(torch.cuda.memory_summary(device=torch.cuda, abbreviated=True))
	
	res_param = {'cur_step_g': cur_step, 'cur_rho1': cur_rho1, 'cur_rho2': cur_rho2, 'cur_rho3': cur_rho3, 'cur_rho4': cur_rho4}
	return G, res_param

def update_HW_Mask(i, model, images, target_label, epsilon, B, init_params, H, noise_Weight, out_iter):
	# initialize learning rate
	cur_step = init_params['cur_step_g']
	cur_rho1 = init_params['cur_rho1']
	cur_rho2 = init_params['cur_rho2']
	cur_rho3 = init_params['cur_rho3']
	cur_rho4 = init_params['cur_rho4']
	
	# initialize y1, y2 as all 1 matrix, and z1, z2, z4 as all zeros matrix
	z1 = torch.zeros_like(B)
	z2 = torch.zeros_like(B)
	z4 = torch.zeros(1).cuda()
	ones = torch.ones_like(B)
	
	for cur_iter in range(1, int(args.maxIter_g)+1):
		B.requires_grad = True
		epsilon.requires_grad = False
		
		# print(torch.cuda.memory_summary(device=torch.cuda, abbreviated=True))
		
		# 1.update y1 & y2
		y1 = torch.clamp((B.detach()+z1/cur_rho1), 0.0, 1.0)  # box constraint (clumped operations???)
		y2 = project_shifted_lp_ball(B.detach()+z2/cur_rho2, 0.5*ones)  # L2 constraint (hardware overhead???)
		
		# 3.update G
		G = torch.sum(torch.multiply(B, H), 0, keepdim=True)
		image_s = images+torch.mul(G, epsilon)
		
		prediction = model(image_s)
		# if(torch.argmax(prediction,1).data==target_label):
		# 	break
		if args.loss == 'ce':
			ce = nn.CrossEntropyLoss()
			loss = ce(prediction, target_label)
		
		elif args.loss == 'cw':
			label_to_one_hot = torch.tensor([[target_label.item()]])
			label_one_hot = torch.zeros(1, args.categories).scatter_(1, label_to_one_hot, 1).cuda()
			
			real = torch.sum(prediction*label_one_hot)
			other_max = torch.max((torch.ones_like(label_one_hot).cuda()-label_one_hot)*prediction-(label_one_hot*10000))
			loss = torch.clamp(other_max-real+args.confidence, min=0)
		
		### update grads
		if B.grad is not None:  # the first time there is no grad
			B.grad.data.zero_()
		loss.backward()
		cnn_grad_B = B.grad
		
		part_1 = 0*2*torch.sum(epsilon*epsilon*torch.sum(B*HW_Map, (0), keepdim=True)*HW_Map, (1, 2, 3), keepdim=True)
		part_2 = args.lambda1*cnn_grad_B
		part_3 = cur_rho1*(B-y1)+z1
		part_4 = cur_rho1*(B-y2)+z2
		part_5 = (cur_rho4*np.max([(B.sum().item()-args.k), 0]))*ones+z4*ones
		
		grad_B = part_1+part_2+part_3+part_4+part_5
		
		B = B-cur_step*grad_B
		B = B.detach()
		
		# # 4.update z1,z2,z3,z4
		z1 = z1+cur_rho1*(B-y1)
		z2 = z2+cur_rho2*(B-y2)
		z4 = z4+cur_rho4*(np.max([(B.sum().item()-args.k), 0]))
		
		# 5.updating rho1, rho2, rho3, rho4
		if cur_iter%args.rho_increase_step == 0:
			cur_rho1 = min(args.rho_increase_factor*cur_rho1, args.rho1_max)
			cur_rho2 = min(args.rho_increase_factor*cur_rho2, args.rho2_max)
			cur_rho4 = min(args.rho_increase_factor*cur_rho4, args.rho4_max)
		
		# updating learning rate
		if cur_iter%args.lr_decay_step == 0:
			cur_step = max(cur_step*args.lr_decay_factor, args.lr_min)
		
		if cur_iter%args.tick_loss_g == 0:
			cur_meta = compute_loss_statistic(model, images, target_label, epsilon, G, args, img_normalized_ops, B, noise_Weight)
			noise_label, _ = compute_predictions_labels(model, images, epsilon, G, args, img_normalized_ops)
		
		# print(torch.cuda.memory_summary(device=torch.cuda, abbreviated=True))
		cur_iter = cur_iter+1
	
	res_param = {'cur_step_g': cur_step, 'cur_rho1': cur_rho1, 'cur_rho2': cur_rho2, 'cur_rho3': cur_rho3, 'cur_rho4': cur_rho4}
	return B, res_param


if __name__ == '__main__':
	main()
	
	g = 0
