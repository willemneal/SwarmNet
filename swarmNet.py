from random import random, uniform, paretovariate
from math import pi, cos, sin, sqrt, isnan

from collections  import Counter
from mpl import AnimatedScatter
from timer import Timer
import numpy as np
from multiprocessing import Pool
from tree import BallTree
from graph import Graph
import matplotlib.pyplot as plt
from functools import partial
from itertools import product

def singleForce(points,a=1,b=1):
    xj, xi = np.array(points[0]), np.array(points[1])
    diff = xj-xi
    if (np.linalg.norm(diff) == 0):
        return np.array([0, 0])
    x =  b * diff/np.linalg.norm(diff)**2 - a * diff
    return x

def sumForces(points,length):
    if len(points) == 0:
        return np.array([0, 0])
    #a,b = Swarm.a,Swarm.b
    t = map(singleForce, points)
    x = np.sum(np.array(t), 0)
    x = x / length
    assert np.shape(x) == (2,)
    return x

class Swarm(list):
    record = False
    a = 1
    b = 1
    TimeStep = .05
    def  __init__(self, memory=None, *args, **kwargs):
        self.tree = None
        self.dt = 1
        self.G = Graph(memory)
        self.alpha = 100000
        self.p = Pool(10)

    def createNew(self, numNodes):
        self.unitConnectionRadius = 5#.15# 1. #/ numNodes**2
        self.walkers = np.array(np.random.uniform(size=(numNodes, 2)))
        self.buildTree()
        self.len = np.size(self.walkers, 0)
        self.G.add_nodes_from(range(numNodes))

    def getSwarm(self):
        ''' list of walkers'''
        return self.getWalkersX(),self.getWalkersY(),self.G.getConnectionNum()

    def timeStep(self,steps=1):

        for step in range(steps):
            self.initframe()
            self.updateGraph()
        return self.walkers,self.G.getConnectionNum()

    def initframe(self):
        self.addForce()
        #self.boundaryCondition()
        self.buildTree()

    def addForce(self):
        dx = Swarm.TimeStep * (self.connectionForce() + self.randomWalk())
        self.walkers += dx

    @staticmethod
    def _map(args,kwargs):
        return np.array(map(args, kwargs))

    def boundaryCondition(self):
        self.walkers = Swarm._map(lambda elm: np.mod(elm, np.ones(2)), self.walkers)

    def getConnections(self):
        #return self.tree.getConnections()
        return np.array([[(self.walkers[x],self.walkers[y]) for y in nodes] for x,nodes in enumerate(self.G.getEdges())])

    def connectionForce(self):
        connections = self.getConnections()
        if len(connections) != 0:
            return np.array(Swarm._map(partial(sumForces,length=self.len), connections))
        return np.repeat([0, 0], self.len, axis=0)




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
        return 0
        angle = self.getRandomAngles()
        direction = np.c_[np.cos(angle),np.sin(angle)]
        return direction * self.getVelocity()

    def getVelocity(self):
        '''return random sample from pareto distrobution'''
        return np.sqrt(np.random.pareto(self.alpha, (self.len, 1)))

    def record(self):
        self.recordWalkers()

    def setPlotData(self):
        self.walkersPlot.set_data(self.getWalkersX(), self.getWalkersY())

    def getWalkersX(self):
        return self.walkers[:,0]

    def getWalkersY(self):
        return self.walkers[:,1]


    def recordWalkers(self, save = False, frames=6000, interval = 100):
        self.ani = AnimatedScatter(self.len,        self.timeStep,
                                   self.getSwarm,   save=save)
        self.ani.show()

    def plotGraphConnections(self):
        plt.figure()
        counts = Counter(self.G.getConnectionNum())
        plt.bar(counts.keys(), counts.values())#self.G.getConnectionNum())
        plt.show()

    def minDist(self):
        return min( [np.linalg.norm(a-b) for a,b in product(self.walkers,repeat=2) if np.linalg.norm(a-b)>0])

    @staticmethod
    def getMins(sizes):
        dists = []
        S = Swarm()
        for s in sizes:
            S.createNew(s)
            S.timeStep(30)
            dists.append(S.minDist())
        return zip(sizes,dists)



if __name__ == "__main__":
    S = Swarm()
    S.createNew(400)
    # #
    S.timeStep(30)
    #dists = Swarm.getMins(range(100,1100,100))
    S.recordWalkers()
    #S.plotGraphConnections()
