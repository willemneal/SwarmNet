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
    def __init__(self,walkers,radius):
        self.walkers = walkers
        self.tree    = BT(walkers)
        self.radius  = radius

    def getEdges(self):
        x = np.array([[(i,p) for p in points] for i, points in enumerate(self.tree.query_radius(self.walkers,self.radius))])
        lst = []
        for point in x:
            lst.extend(point)
        return np.array(lst)


    def getConnections(self):
        results = range(np.size(self.walkers,0))

        # print self.tree.query_radius(self.walkers,self.radius)
        for i,neighbors in enumerate(self.tree.query_radius(self.walkers,self.radius)):
            #print neighbors
            results[i] = np.array([np.array([self.walkers[i],self.walkers[n]]) for n in neighbors if i != n])
            #print points
        #print results

        return np.array(results)

def NearBy():
    def __init__(self,walkers):
        self.walkers = walkers
        self.dim = 2

class Swarm(list):
    record = False
    a = 1
    b = .1
    TimeStep = .01
    def  __init__(self, *args, **kwargs):
        self.tree = None
        self.dt = 1
        self.G = nx.Graph()
        #self.a = 2
        self.alpha = 100000
        self.p = Pool(10)






    def createNew(self, numNodes):
        self.unitConnectionRadius = 1./numNodes**2
        self.walkers = np.array(np.random.uniform(size=(numNodes,2)))
        self.buildTree()
        self.len = np.size(self.walkers,0)

        self.G.add_nodes_from(range(numNodes))





    def getVelocity(self):
        '''return random sample from pareto distrobution'''

        return np.sqrt(np.random.pareto(self.alpha,(self.len,1)))


    def getSwarm(self):
        ''' list of walkers'''
        return self.walkers

    def timeStep(self,steps=1):
        self.initframe()
        #self.updateGraph()

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
        return np.array(map(args,kwargs))

    def boundaryCondition(self):
        self.walkers = Swarm._map(lambda elm: np.mod(elm,np.ones(2)),self.walkers)

    def getConnections(self):
        return self.tree.getConnections()
        return np.array([[(self.walkers[i],self.walkers[j]) for j in nx.all_neighbors(self.G, i)] for i in range(self.len)])



    def connectionForce(self):

        connections = self.getConnections()# self.tree.getEdges(self.unitConnectionRadius)

        if len(connections) != 0:

            return Swarm._map(self.sumForces, connections)

        return np.repeat([0,0],self.len,axis=0)


    def sumForces(self, points):


        if len(points)==0:
            return np.array([0,0])
    #    print np.size(points,0)
        #print np.sum(np.array([Swarm.singleForce(point) for point in points]))
        t = [Swarm.singleForce(point) for point in points]
        x = np.sum(np.array(t),0)
        #print x, len(points)
        x = x/self.len
        assert np.shape(x) == (2,)
        return x

    @staticmethod
    def singleForce(points):
    #    print np.size(points), type(points)
        xj, xi = np.array(points[0]),np.array(points[1])
        diff = xj-xi
        #print diff/np.linalg.norm(diff)**2
        if (diff == np.array([0,0])).all():
            return np.array([0,0])
        x =  Swarm.b*diff/np.linalg.norm(diff)**2 - Swarm.a*diff
        #print x
        return x


    def buildTree(self):
        ''' Kdtree to search for neighbors '''
        self.tree = BallTree(self.walkers,self.unitConnectionRadius)

    def updateGraph(self):
        self.G.add_edges_from(self.tree.getEdges())

    def getWalkersLocation(self):
        return [walker.location() for walker in self]

    def getRandomAngles(self):
        return np.random.uniform(low=0,high=2*pi,size=(self.len,1))



    def randomWalk(self):
        angle = self.getRandomAngles()
        direction = np.c_[np.cos(angle),np.sin(angle)]

        return direction * self.getVelocity()


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

    def recordWalkers(self, save = False, frames =1200, interval = 100):
        self.initRecord()
        ani = animation.FuncAnimation(self.fig, self.timeStep, frames,
                                      interval=100)
        if save: ani.save('%d-V2.mp4' % (self.len), fps=60, extra_args=['-vcodec', 'libx264'])
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
    #S.plotGraphConnections()
