from random import random, uniform
from math import pi, cos, sin, sqrt
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
        u = random()+random()
        self.r = u if u<=1 else 2-u
        self.t = 2*pi*random()
        self.x =  self.r*cos(self.t)
        self.y =  self.r*sin(self.t)
        self.Vin = 1./100.
        self.Vint= uniform(0,2*pi)

    def next(self):
        newAngle = uniform(0,2*pi)
        self.x += self.Vin*cos(self.Vint + newAngle)
        self.y += self.Vin*sin(self.Vint + newAngle)

    def dist(self,other):
        return sqrt((self.x-other.x)**2 + (self.y-other.y)**2)

    def bumped(self,other):
            return self.dist(other)<=unitConnectionRadius

    def __repr__(self):
        return "(%.4f,%.4f)"%(self.x,self.y)

    def location(self):
        return (self.x,self.y)

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
        with Timer() as t:
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
        map(lambda x: x.next(),self)

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
                             xlim=(-1, 1), ylim=(-1, 1))
        self.walkersPlot, = self.ax.plot(self.getWalkersX(), self.getWalkersY(), 'bo',ms=2)

    def recordWalkers(self, save=False, frames = 1, interval = 100):
        self.initRecord()
        ani = animation.FuncAnimation(self.fig, self.timeStep, frames,
                                      interval=100)
        if save: ani.save('walkers-%5d.mp4'%(len(self)), fps=30, extra_args=['-vcodec', 'libx264'])
        plt.show()

    def plotGraphConnections(self):
        counts = Counter(self.G.degree().values())
        plt.figure()
        plt.bar(counts.keys(),counts.values())
        plt.show()

S = Swarm()
S.createNew(100000)
S.recordWalkers()
