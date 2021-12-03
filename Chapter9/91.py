from matplotlib.pyplot import figure
import networkx as nx

# gr = nx.DiGraph()
# gr.add_nodes_from([0, 1, 2, 3])

# gr.add_edge(0, 1, capacity=4)
# gr.add_edge(1, 2, capacity=3)
# gr.add_edge(2, 3, capacity=5)
# gr.add_edge(0, 2, capacity=3)
# gr.add_edge(1, 3, capacity=4)

# flows, dic = nx.maximum_flow(gr, 0, 3)
# cuts, par = nx.minimum_cut(gr, 0, 3)
# print('flow is:', dic)
# print('cut is:', par)
# for label, poss in enumerate(par):
#     for pos in poss:
#         print(pos, label)

## Test segmentation using graph cut
import graphcut
import numpy as np
from PIL import Image
from pylab import *

im = Image.open('empire.jpg')
size = tuple((np.array(im.size) * 0.07).astype(int))
im = np.array(im.resize(size, Image.BILINEAR))
size = im.shape[:2]

figure()
imshow(im)

# add two retangular training regions
labels = np.zeros(size)
labels[3:18, 3:18] = -1
labels[-18:-3, -18:-3] = 1

# create graph
g = graphcut.build_bayes_graph(im, labels, kappa=3)

# cut the graph
res = graphcut.cut_graph(g, size)

figure()
graphcut.show_labeling(im, labels)

figure()
imshow(res)
gray()
# axis('off')

show()


