import networkx as nx

import numpy as np
from memory import Memory
class Graph(nx.Graph):
    def __init__(self,memory=None):
        super(Graph,self).__init__()
        self.M = Memory(memory)

    def add_edges_from(self,edges):
        super(Graph,self).add_edges_from(edges)
        self.remove_edges_from(self.M.addExp(edges))

    def getConnectionNum(self):
        return np.array([self.degree(node) for node in range(self.number_of_nodes())])

    def getEdges(self):
        return np.array([nx.all_neighbors(self, node) for node in range(self.number_of_nodes())])
