import sys
from networkx.algorithms.flow.maxflow import maximum_flow
sys.path.insert(0, r'G:/dvtu/ThucTap/Chapter8')
import bayes
import networkx as nx
import numpy as np
from pylab import *

def build_bayes_graph(im, labels, sigma=1e2, kappa=2):
    m, n = im.shape[:2]

    # RGB vector version (one pixel per row)
    vim = im.reshape((-1, 3))

    # RGB for foreground and background
    foreground = im[labels==1].reshape((-1, 3))
    background = im[labels==-1].reshape((-1, 3))
    train_data = [foreground, background]

    # train naive Bayes classifier
    bc = bayes.BayesClassifier()
    bc.train(train_data)

    # get probabilities for all pixels
    bc_labels, prob = bc.classify(vim)
    prob_fg = prob[0]
    prob_bg = prob[1]

    # create graph with m*n + 2 nodes
    gr = nx.DiGraph()
    gr.add_nodes_from(range(m * n + 2))

    source = m*n # second to last is source
    sink = m*n + 1

    # normalize
    for i in range(vim.shape[0]):
        vim[i] = vim[i] / np.linalg.norm(vim[i])
    
    # go through all nodes and add edges
    for i in range(m*n):
        # add edge from source
        gr.add_edge(source, i, capacity=(prob_fg[i]/(prob_fg[i] + prob_bg[i])))

        # add edge to sink
        gr.add_edge(i, sink, capacity=(prob_bg[i]/(prob_fg[i] + prob_bg[i])))

        # add edges to neighbors
        if i % n != 0: # left exists
            edge_wt = kappa * np.exp(-1.0 * np.sum((vim[i] - vim[i-1])**2)/sigma)
            gr.add_edge(i, i - 1, capacity=edge_wt)
        if (i + 1) % n != 0: # right exists
            edge_wt = kappa * np.exp(-1.0 * np.sum((vim[i] - vim[i + 1])**2)/sigma)
            gr.add_edge(i, i + 1, capacity=edge_wt)
        if i//n != 0: # up exists
            edge_wt = kappa * np.exp(-1.0 * np.sum((vim[i] - vim[i - n])**2)/sigma)
            gr.add_edge(i, i - n, capacity=edge_wt)
        if i//n != m-1: # down exists
            edge_wt = kappa * np.exp(-1.0 * np.sum((vim[i] - vim[i + n])**2)/sigma)
            gr.add_edge(i, i + n, capacity=edge_wt)

    return gr

def show_labeling(im, labels):
    """ Show image with foreground and background areas
        labels = 1 for foreground, -1 for background, 0 otherwise. """
    imshow(im)
    contour(labels, [-0.5, 0.5])
    contourf(labels, [-1, -0.5], colors='b', alpha=0.25)
    contourf(labels, [0.5, 1], colors='r', alpha=0.25)
    axis('off')

def cut_graph(gr, imsize):
    """ Solve max flow of graph gr and return binary
        labels of the resulting segmentation."""
    m, n = imsize
    source = m * n
    sink = m * n + 1

    # cut the graph
    cuts, par = nx.minimum_cut(gr, source, sink)

    print(len(par))

    # convert graph to image with labels
    res = np.zeros(m * n)
    for label, poss in enumerate(par):
        for pos in poss:
            if pos != source and pos != sink:
                res[pos] = label
    print(res)
    return res.reshape((m, n))

