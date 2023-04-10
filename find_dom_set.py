import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.approximation as nxaa
from collections import OrderedDict, deque
import copy
import operator
import sys
import numpy as np
import os
import random
random.seed(123)
import math
#from sinkhorn_knopp import sinkhorn_knopp as skp

def find_vertices_edges(connectivity):
	vertices = [i for i in range(len(connectivity))]
	edges = []
	for i in range(len(connectivity)):
		if i+1 < len(connectivity):
			for j in range(i+1, len(connectivity)):
				if connectivity[i][j] == 1:
					edges.append((i,j))
	return vertices, edges

def create_graph_netwrokx(vertices, edges):
	G = nx.Graph()
	G.add_nodes_from(vertices)
	G.add_edges_from(edges)
	#print("Vertices: ", G.nodes())
	#print("Edges: ", G.edges())
	return G

class DominatingSets:
	@classmethod
	def get_dominating_sets(cls, G, weight=None):
		"""get a dominating sets 
		"""
		dominating_sets = nxaa.min_weighted_dominating_set(G, weight=weight)

		return dominating_sets

	@classmethod
	def min_connected_dominating_sets_non_distributed(cls, G):
		"""Compute a CDS, based on algorithm of Butenko, Cheng, Oliveira, Pardalos
			Based on the paper: BUTENKO, Sergiy, CHENG, Xiuzhen, OLIVEIRA, Carlos A., et al. A new heuristic for the minimum connected dominating set problem on ad hoc wireless networks. In : Recent developments in cooperative control and optimization. Springer US, 2004. p. 61-73.
		"""
		assert nx.is_connected(G) 

		G2 = copy.deepcopy(G)
		#for n,degree in G2.degree:
		#	print(degree)

		# Step 1: initialization
		# take the node with maximum degree as the starting node
		starting_node = max(G2.degree(), key=operator.itemgetter(1))[0] 
		fixed_nodes = {starting_node}

		# Enqueue the neighbor nodes of starting node to Q in descending order by their degree
		neighbor_nodes = G2.neighbors(starting_node)
		neighbor_nodes_ls = list(neighbor_nodes)

		neighbor_nodes_sorted = OrderedDict(sorted(G2.degree(neighbor_nodes_ls), key=operator.itemgetter(1), reverse=True)).keys()
		
		priority_queue = deque(neighbor_nodes_sorted) # a priority queue is maintained centrally to decide whether an element would be a part of CDS.
		
		inserted_set = set(list(neighbor_nodes_sorted) + [starting_node])

		# Step 2: calculate the cds
		while priority_queue:
			u = priority_queue.pop()

			# check if the graph after removing u is still connected
			rest_graph = copy.deepcopy(G2)
			rest_graph.remove_node(u)

			if nx.is_connected(rest_graph):
				G2.remove_node(u)

			else: # is not connected 
				fixed_nodes.add(u)

				# add neighbors of u to the priority queue, which never are inserted into Q
				inserted_neighbors = set(G2.neighbors(u)) - inserted_set
				inserted_neighbors_sorted = OrderedDict(sorted(G2.degree(inserted_neighbors),
																key=operator.itemgetter(1), reverse=True)).keys()

				priority_queue.extend(inserted_neighbors_sorted)
				inserted_set.update(inserted_neighbors_sorted)

		# Step 3: verify the result
		assert nx.is_dominating_set(G, fixed_nodes) and nx.is_connected(G.subgraph(fixed_nodes))

		return fixed_nodes

def vis_DS(graph, folder_name, ds):

	color_map = []
	for node in graph:
		if node in ds:
			color_map.append("red")
		else:
			color_map.append("blue")
	pos = nx.circular_layout(graph)
	nx.draw(graph, pos, cmap=plt.get_cmap('viridis'), node_color=color_map, with_labels=True, font_color='white')
	plt.savefig(f"{folder_name}/MCDS_fig.png")

def find_dom(connectivity, save_folder, MDS = False, MCDS = False):

	vertices, edges = find_vertices_edges(connectivity)
	graph = create_graph_netwrokx(vertices, edges)
	if MCDS:
		mcds = DominatingSets.min_connected_dominating_sets_non_distributed(graph)
		#print("Minimum connected dominating set =", mcds)
		vis_DS(graph, save_folder, mcds)
		return mcds
	if MDS:
		mds = DominatingSets.get_dominating_sets(graph)
		#print("Minimum dominating set =", mds)
		vis_DS(graph, save_folder, mds)
		return mds

def gen_connect_dom(DS, connectivity):

	vertices, edges = find_vertices_edges(connectivity)
	connectivity_dom = [[0 for _ in range(len(DS))] for _ in range(len(DS))]
	pi = [[0 for _ in range(len(DS))] for _ in range(len(DS))]
	for i, dom_node in enumerate(DS):
		connectivity_dom[i][i] = 1
		#pi[i][i] = 1
		for j, dom_node_2 in enumerate(DS):
			if (dom_node, dom_node_2) in edges or (dom_node_2, dom_node) in edges:
				connectivity_dom[i][j] = 1
				connectivity_dom[j][i] = 1
				#pi[i][j] = 1
				#pi[j][i] = 1

	num_con_neigh = [connectivity_dom[i].count(1) for i in range(len(DS))]
	max_con = max(num_con_neigh)
	pi = np.array(connectivity_dom)/max_con

	for i in range(len(DS)):
		pi[i][i] = 1 - (num_con_neigh[i]-1)*(1/max_con)

	return connectivity_dom, pi

def find_PI(connectivity,pi_dom,DS):

	PI_MCD = np.zeros((len(connectivity),len(connectivity)))
	len_dom = len(pi_dom)

	for i, dom_node_i in enumerate(DS):
		for dom_node_j in DS:
			if connectivity[dom_node_i][dom_node_j] == 1 and dom_node_i != dom_node_j:
				connectivity[dom_node_i][dom_node_j] = 0
				connectivity[dom_node_j][dom_node_i] = 0
		PI_MCD[dom_node_i][dom_node_i] = pi_dom[i][i]

	for dom_node_i in DS:
		num_con_nondom = connectivity[dom_node_i].count(1)
		pi_ii = PI_MCD[dom_node_i][dom_node_i]
		for j, elm_con in enumerate(connectivity[dom_node_i]):
			if elm_con == 1:
				PI_MCD[dom_node_i][j] = pi_ii/num_con_nondom

	#make matrix symmetric
	PI_MCD = np.maximum(PI_MCD, PI_MCD.transpose())


	for i, dom_node_i in enumerate(DS):
		for j, dom_node_j in enumerate(DS):
			if dom_node_i != dom_node_j:
				PI_MCD[dom_node_i][dom_node_j] = pi_dom[i][j]
				PI_MCD[dom_node_j][dom_node_i] = pi_dom[j][i]

	for i, elm_pi_i in enumerate(PI_MCD):
		if PI_MCD[i][i] == 0:
			PI_MCD[i][i] = 1-sum(PI_MCD[i])

	return PI_MCD





	
