import os, os.path
import json
import glob
import argparse
import matplotlib.pylab as plt
import numpy as np
import torch
from collections import OrderedDict
import torch.backends.cudnn as cudnn

def movingaverage (values, window):
	weights = np.repeat(1.0, window)/window
	sma = np.convolve(values, weights, 'valid')
	return sma

def plot_results(graph_type, n_agent, datashuffle, dataset, model_name,list_experiments,type_of_plot,plot_upper_bound,moving_avg_win):

	n_ag = 10
	list_color = [u'#FF6600',# orange
				  u'#00FF00',# green
				  u'#660099',# purple
				  u'#FF0000',# red
				  u'#0000FF',# blue
				  u'#9900CC',# violet
				  u'#FFFF00']# yellow

	graph_log = []
	agent_log = []
	exp_log = []

	# get experiment names
	graph_names = []
	omega_list = []
	beta_list = []
	experiment_name = {'CDSGD':'CDSGD', 'LGA' : 'CGA', 'SGP' : 'SGP', 'SwarmSGD' : 'SwarmSGD', 'CDMSGD':'DPMSGD', 'CompLGA' : 'CompLGA', 'DSMA':'DSMA', 'DSMA_pi':'DSMA_pi', 'Matcha':'Matcha', 'LDSGD':'LDSGD'}
	graph_name = {'FC':'Fully-Connected', 'Ring':'Ring', 'Bipar':'Bipartite'}
	title_n_ag = {'5 agents':'5 agents', '10 agents':'10 agents', '40 agents':'40 agents'}
	title_model_name = {'CNN':'CNN', 'LR':'LR', 'resnet20':'ResNet20', 'VGG11': 'VGG11'}
	

	if 1 in type_of_plot:
		for graph in graph_type:
			for n in n_agent:
				n_agent_name = str(n)+'_agents_all'
				exp_log.append(glob.glob(os.path.join("log",n_agent_name,datashuffle,dataset,model_name,graph,list_experiments,"global.json")))
		#n_agent_proof_name = str(n)+'_agents_proof'
		#exp_log.append(glob.glob(os.path.join("log",n_agent_name,datashuffle,dataset,model_name,graph,list_experiments,"global.json")))
	
	if 2 in type_of_plot:
		n_agent_name = str(n_agent[0])+'_agents_all'
		for graph in graph_type:
			for exp in list_experiments:
				exp_log.append(glob.glob(os.path.join("log",n_agent_name,datashuffle,dataset,model_name,graph,exp,"global.json")))
		#n_agent_proof_name = str(n_agent[0])+'_agents_proof'
		#exp_log.append(glob.glob(os.path.join("log",n_agent_proof_name,datashuffle,dataset,model_name,graph,'DSMA_0.9',"global.json")))
	####################################################### plotting the graph_avg_acc #############################################
	if 1 in type_of_plot:
		print("Plotting the graph_avg_ACC...")

		plt.figure()
		for idx, log in enumerate(exp_log):
			model_name_h = model_name[idx]
			print(model_name_h)
			with open(log[0], "r") as f:
				d = json.load(f)
				color = list_color[idx]
				targ_data = d["trainload_acc"][:plot_upper_bound]
				targ_data = movingaverage(targ_data,moving_avg_win)
				
				Label = title_model_name[model_name_h]				
				plt.plot(targ_data,
						 color=color,
						 linewidth=2,
						 label=Label)
				targ_data = d["testloader_acc"][:plot_upper_bound]
				targ_data = movingaverage(targ_data,moving_avg_win)
				plt.plot(targ_data,
						 color=color,
						 linewidth=2,
						 linestyle=":",)
		plt.ylabel("Accuracy(%)", fontsize=14)
		plt.xlabel("Number of epochs", fontsize=14)
		plt.ylim([0,100])
		plt.legend(loc="best",fontsize=12)
		plt.tight_layout()

		plt.savefig("./figures/paper/resnet/%s_%s_%s_%s_%s.pdf" % (graph_type[-1],datashuffle,dataset,model_name, "avg_acc_n_agent"))
		plt.show()


	####################################################### plotting the graph_avg_acc #############################################
	if 2 in type_of_plot:
		print("Plotting the graph_avg_ACC...")

		plt.figure()
		list_experiments.append('DSMA_0.9')
		print(exp_log)
		for idx, log in enumerate(exp_log):
			temp = list_experiments[idx].split('_')
			opt_name = '_'.join(temp[:-1])
			with open(log[0], "r") as f:
				d = json.load(f)
				color = list_color[idx]
				#targ_data = d["trainload_acc"][:plot_upper_bound]
				#targ_data = movingaverage(targ_data,moving_avg_win)
				
				Label = experiment_name[opt_name]
				
				#plt.plot(targ_data,
				#		 color=color,
				#		 linewidth=2,
				#		 label=Label)
				targ_data = d["testloader_acc"][:plot_upper_bound]
				#targ_data = d["train_loss"][:plot_upper_bound]
				targ_data = movingaverage(targ_data,moving_avg_win)
				plt.plot(targ_data,
						 color=color,
						 linewidth=2,
						 label=Label)
						 #linestyle=":",)
		plt.ylabel("Accuracy(%)", fontsize=24)
		plt.xlabel("Number of epochs", fontsize=24)
		plt.xlim([0,plot_upper_bound])
		plt.xticks(fontsize=18)
		plt.yticks(fontsize=18)
		plt.ylim([0,100])
		#plt.legend(loc="best",fontsize=12)
		plt.tight_layout()

		plt.savefig("./figures/final/acc/supmnist/%s_%s_%s_%s_%s_%s.pdf" % (n_agent[0],graph_type[-1],dataset,datashuffle,model_name, "avg_test_acc_comp"))
		plt.show()






if __name__ == '__main__':


	parser = argparse.ArgumentParser(description='Plot results of experiments')
	parser.add_argument('graph_type', type=str, nargs='+',
					help='FC, Ring, Bipar')
	parser.add_argument('-opt','--list_experiments', type=str, nargs='+',
					help='list_experiments')				
	parser.add_argument('-d','--dataset', type=str,default="CIFAR10",
						help='name of the dataset: cifar10, cifar100 or mnist')
	parser.add_argument('-ds','--datashuffle', type=str,default="iid",
						help='data shuffle: non-iid, or iid')
	parser.add_argument('-m','--model_name', type=str, default="CNN",
					help='Model name')
	parser.add_argument('-u','--plot_upper_bound', type=int, default=300,
					help='Upper bound for plotting. Eg: 100 = plot 0 ~ 100 iterations ')
	parser.add_argument('-win','--moving_avg_win', type=int, default=3,
					help='moving_avg_win')
	parser.add_argument('-top','--type_of_plot', type=int, nargs='+', default=1,
					help='Type of plot example: 1 2 3, ...')
	parser.add_argument('-n_ag','--number_agent', type=int, nargs='+', default=5,
				help='number of agents')

	'''
	Type_of_plot:
	1: graph_avg_loss
	2: graph_avg_acc
	3: hist of param for each graph TYPE
	4: hist of param for each graph AGENT
	5: gradients
	'''

	args = parser.parse_args()
	dataset = args.dataset
	list_experiments = args.list_experiments
	model_name = args.model_name
	datashuffle = args.datashuffle
	plot_upper_bound = args.plot_upper_bound
	graph_type = args.graph_type
	n_agent = args.number_agent
	moving_avg_win = args.moving_avg_win
	type_of_plot = args.type_of_plot

	assert max(type_of_plot)<6 and min(type_of_plot)>0, "type_of_plot should be in [1,2,3,4,5]"

	plot_results(graph_type, n_agent,datashuffle,dataset,model_name,list_experiments,type_of_plot,plot_upper_bound,moving_avg_win)


'''

/ocean/projects/cis200022p/shared/conda_pytorch_dist/bin/python comp_plot.py FC -n_ag 5 -opt LGA_0.9 CDMSGD_0.9 SGP_0.9 SwarmSGD_0.9 -d CIFAR10 -m CNN -u 150 -win 3 -ds non-iid -top 2


'''