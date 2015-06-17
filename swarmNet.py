from random import random, uniform
from math import pi, cos, sin, sqrt, isnan
import networkx as nx
from collections  import Counter
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import spatial as sp
from timer import Timer
from sklearn.neighbors import BallTree as BT
import numpy as np
WALKERS = 100

class Walker:
    number=0
    def __init__(self, *args, **kwargs):
        self.num = Walker.number
        Walker.number +=1
        self.initialPosition()
        self.initialVelocity()
        self.a = 1000000
        self.markerSize = 4


    def cicrleDistribution(self):
        u = random()+random()
        self.r = u if u<=1 else 2-u
        self.t = 2*pi*random()
        self.x =  self.r*cos(self.t)
        self.y =  self.r*sin(self.t)

    def initialVelocity(self,fixed=1./100):
        self.fixedVelocity()

    def randomVelocity(self):
        pass

    def fixedVelocity (self):
        self.Vin = 1./100.

    def initalAngle(self):
        self.Vint= uniform(0,2*pi)

    def initialPosition(self):
        self.y, self.x = (uniform(-1,1) for i in range(2))
        self.initalAngle()


    def connectionForce(self, graph):
        val = 0
        for node in nx.all_neighbors(graph, self):
            val += self.calculateForce(node)
        res = val / self.getNorm(graph)
        if isnan(res[0]) or isnan(res[1]):
            return 0
        return res


    def getNorm(self, graph):
        return nx.number_of_nodes(graph)
        return self.numNeighbors(graph)

    def calculateForce(self,other):
        diff = np.array(self-other)
        card = abs(diff[0]**2 + diff[1]**2)
        repulsion = 0 if (card == 0) else diff/card
        return repulsion - self.a*diff

    def next(self,graph):
        newAngle = uniform(0,2*pi)
        self.x += self.Vin*cos(self.Vint + newAngle)
        self.y += self.Vin*sin(self.Vint + newAngle)
        if self.numNeighbors(graph) > 0:
            self.addForce(self.connectionForce(graph))
        self.checkBounary()





    def checkBounary(self):
        self.x = self.x % 1
        self.y = self.y % 1

    def addForce(self,vector):
        self.x += vector[0]
        self.y += vector[1]

    def numNeighbors(self,graph):
        return nx.degree(graph,self)

    def dist(self,other):
        return sqrt((self.x-other.x)**2 + (self.y-other.y)**2)

    def bumped(self,other):
            return self.dist(other) <= unitConnectionRadius

    def __repr__(self):
        return "(%.4f,%.4f)"%(self.x,self.y)

    def location(self):
        return (self.x,self.y)

    def __sub__(self,other):
        return (self.x-other.x,self.y-other.y)

    @staticmethod
    def createSwarm(number, *args, **kwargs):
        '''
        Little note: one would think that[Walker()]*5
            would work, but this copies the same pointer to the object
            However, in the case of arguments this is not a problem.

        '''
        return map(Walker, [ (args, kwargs) ] * number)
        return [Walker(args,kwargs) for i in range(number)]



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
    def __init__(self,walkers):
        self.walkers = walkers
        self.tree = BT(walkers.getWalkersLocation())

    def getEdges(self, radius):
        results = []
        for i,neighbors in enumerate(self.tree.query_radius(self.walkers.getWalkersLocation(),radius)):
            if len(neighbors) > 0:
                for n in neighbors:
                    results.append((self.walkers[i],self.walkers[n]))
        return results

def NearBy():
    def __init__(self,walkers):
        self.walkers = walkers
        self.dim = 2

class Swarm(list):
    record = False
    def  __init__(self, *args, **kwargs):
        self.tree = None
        self.dt = 1
        self.G = nx.Graph()


    def createNew(self, numNodes):
        self.extend(Walker.createSwarm(numNodes))
        self.G.add_nodes_from(self)
        self.unitConnectionRadius = sqrt(1./numNodes)


    def getSwarm(self):
        ''' list of walkers'''
        return self

    def timeStep(self,steps=1):
        self.initframe()
        self.updateGraph()

    def initframe(self):
        self.setPlotData()
        map(lambda x: x.next(self.G),self)

    def buildTree(self):
        ''' Kdtree to search for neighbors '''
        self.tree = BallTree(self)

    def updateGraph(self):
        self.buildTree()
        self.G.add_edges_from(self.tree.getEdges(self.unitConnectionRadius))

    def getWalkersLocation(self):
        return [walker.location() for walker in self]

    def record(self):
        self.recordWalkers()

    def setPlotData(self):
        self.walkersPlot.set_data(self.getWalkersX(),self.getWalkersY())

    def getWalkersX(self):
        return [walker.x for walker in self]

    def getWalkersY(self):
        return [walker.y for walker in self]

    def initRecord(self):
        self.fig = plt.figure()

        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        self.ax = self.fig.add_subplot(111, aspect='equal', autoscale_on=False,
                             xlim = (0,1), ylim=(0, 1))

        self.walkersPlot, = self.ax.plot(self.getWalkersX(), self.getWalkersY(), 'bo',ms=4)

    def recordWalkers(self, save = False, frames = 10, interval = 100):
        self.initRecord()
        ani = animation.FuncAnimation(self.fig, self.timeStep, frames,
                                      interval=100)
        if save: ani.save('walkers-%5d-Nnormalized.mp4'%(len(self)), fps=1, extra_args=['-vcodec', 'libx264'])
        plt.show()

    def plotGraphConnections(self):
        counts = Counter(self.G.degree().values())
        plt.figure()
        plt.bar(counts.keys(),counts.values())
        plt.show()

S = Swarm()
S.createNew(100)
S.recordWalkers(save=True)
S.plotGraphConnections()
