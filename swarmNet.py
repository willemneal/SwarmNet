from random import random, uniform, paretovariate
from math import pi, cos, sin, sqrt, isnan
import networkx as nx
from collections  import Counter
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import spatial as sp
from timer import Timer
from sklearn.neighbors import BallTree as BT
import numpy as np
from multiprocessing import Pool
from memory import Memory
WALKERS = 100

class Graph(nx.Graph):
    def __init__(self,memory=50):
        super(Graph,self).__init__()
        self.M = Memory(memory)

    def add_edges_from(self,edges):
        super(Graph,self).add_edges_from(edges)
        self.remove_edges_from(self.M.addExp(edges))

class KDTree(sp.kdtree.KDTree):
    def __init__(self,walkers):
        self.walkers = walkers
        super(KDTree,self).__init__([walker.location() for walker in walkers])

    def getEdges(self, radius):
        results = []
        with Timer() as t:
            for i,neighbors in enumerate(self.query_ball_tree(self,radius)):
                if len(neighbors) > 0:
                    for n in neighbors:
                        results.append((self.walkers[i],self.walkers[n]))
        return results

class BallTree():
    def __init__(self,walkers,radius):
        self.walkers = walkers
        self.tree    = BT(walkers)
        self.radius  = radius

    def getEdges(self):
        p = self.tree.query_radius(self.walkers,self.radius)
        #print p
        x = np.array([[(i,p) for p in points if p!=i] for i, points in enumerate(p)])
        lst = []
        for point in x:
            lst.extend(point)
        return np.array(lst)


    def getConnections(self):
        results = range(np.size(self.walkers, 0))
        for i,neighbors in enumerate(self.tree.query_radius(self.walkers, self.radius)):
            results[i] = np.array([np.array([self.walkers[i], self.walkers[n]]) for n in neighbors if i != n])
        return np.array(results)

def NearBy():
    def __init__(self,walkers):
        self.walkers = walkers
        self.dim = 2

class Swarm(list):
    record = False
    a = 1
    b = .1
    TimeStep = .1
    def  __init__(self, *args, **kwargs):
        self.tree = None
        self.dt = 1
        self.G = Graph(100)#nx.Graph()
        self.alpha = 100000
        self.p = Pool(10)

    def createNew(self, numNodes):
        self.unitConnectionRadius = 1. / numNodes
        self.walkers = np.array(np.random.uniform(size=(numNodes, 2)))
        self.buildTree()
        self.len = np.size(self.walkers, 0)
        self.G.add_nodes_from(range(numNodes))

    def getSwarm(self):
        ''' list of walkers'''
        return self.walkers

    def timeStep(self,steps=1):
        self.initframe()
        self.updateGraph()

    def initframe(self):
        self.setPlotData()
        self.addForce()
        self.boundaryCondition()
        self.buildTree()

    def addForce(self):
        dx = Swarm.TimeStep*self.connectionForce() + self.randomWalk()
        self.walkers += dx

    @staticmethod
    def _map(args,kwargs):
        return np.array(map(args, kwargs))

    def boundaryCondition(self):
        self.walkers = Swarm._map(lambda elm: np.mod(elm, np.ones(2)), self.walkers)

    def getConnections(self):
        #return self.tree.getConnections()
        return np.array([[(self.walkers[i],self.walkers[j]) for j in nx.all_neighbors(self.G, i)] for i in range(self.len)])


    def connectionForce(self):
        connections = self.getConnections()
        if len(connections) != 0:
            return Swarm._map(self.sumForces, connections)
        return np.repeat([0, 0], self.len, axis=0)

    def sumForces(self, points):
        if len(points) == 0:
            return np.array([0, 0])
        t = [Swarm.singleForce(point) for point in points]
        x = np.sum(np.array(t), 0)
        x = x / self.len
        assert np.shape(x) == (2,)
        return x

    @staticmethod
    def singleForce(points):
        xj, xi = np.array(points[0]), np.array(points[1])
        diff = xj-xi
        if (diff == np.array([0, 0])).all():
            return np.array([0, 0])
        x =  Swarm.b * diff/np.linalg.norm(diff)**2 - Swarm.a * diff
        return x


    def buildTree(self):
        ''' Kdtree to search for neighbors '''
        self.tree = BallTree(self.walkers, self.unitConnectionRadius)


    def updateGraph(self):
        edges = self.tree.getEdges()
        #print len(nx.edges(self.G))
        self.G.add_edges_from(edges)
        #print nx.edges(self.G)

    def getWalkersLocation(self):
        return [walker.location() for walker in self]

    def getRandomAngles(self):
        return np.random.uniform(low=0, high=2*pi, size=(self.len, 1))

    def randomWalk(self):
        angle = self.getRandomAngles()
        direction = np.c_[np.cos(angle),np.sin(angle)]
        return direction * self.getVelocity()

    def getVelocity(self):
        '''return random sample from pareto distrobution'''
        return np.sqrt(np.random.pareto(self.alpha,(self.len,1)))

    def record(self):
        self.recordWalkers()

    def setPlotData(self):
        self.walkersPlot.set_data(self.getWalkersX(),self.getWalkersY())

    def getWalkersX(self):
        return self.walkers[:,0]

    def getWalkersY(self):
        return self.walkers[:,1]

    def initRecord(self):
        self.fig = plt.figure()
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.ax = self.fig.add_subplot(111, aspect='equal', autoscale_on=False,
                                       xlim = (0,1), ylim=(0, 1))
        self.walkersPlot, = self.ax.plot(self.getWalkersX(), self.getWalkersY(), 'bo',ms=4)

    def recordWalkers(self, save = False, frames=6000, interval = 100):
        self.initRecord()
        ani = animation.FuncAnimation(self.fig, self.timeStep, frames, interval=100)
        if save: ani.save('%d-WithGraphR2.mp4' % (self.len), fps=60, extra_args=['-vcodec', 'libx264'])
        plt.show()

    def plotGraphConnections(self):
        counts = Counter(self.G.degree().values())
        plt.figure()
        plt.bar(counts.keys(),counts.values())
        plt.show()


if __name__ == "__main__":
    S = Swarm()
    S.createNew(100)
    S.recordWalkers(save=True)
    S.plotGraphConnections()
