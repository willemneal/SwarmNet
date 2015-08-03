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


class AnimatedScatter(object):
        def __init__(self, numpoints, update, initState,save=False,fps=60):
            self.numpoints = numpoints
            self.update = update
            self.init = initState
            # Setup the figure and axes...
            self.fig, self.ax = plt.subplots(figsize=(10,10))
            self.ax.set_title("%d walkers"%(self.numpoints))
            # Then setup FuncAnimation.
            self.ani = animation.FuncAnimation(self.fig, self.next,frames=6000, interval=100,
                                               init_func=self.setup_plot, blit=True)
            if save: self.ani.save('%d-Scatter2.mp4' % (self.numpoints), fps=fps, extra_args=['-vcodec', 'libx264'])

        def setup_plot(self):
            """Initial drawing of the scatter plot."""
            x, y, c = self.init()
            self.scat = self.ax.scatter(x, y, c=c, s=120, vmin=0,vmax=30,animated=True,cmap=plt.cm.get_cmap('winter'))
            #plt.colorbar(self.scat)
            self.ax.axis([0, 1, 0, 1])
            # For FuncAnimation's sake, we need to return the artist we'll be using
            # Note that it expects a sequence of artists, thus the trailing comma.
            return self.scat,

        def next(self,i):
            points, c = self.update()
            self.scat.set_offsets(points)
            self.scat.set_array(c)
            self.ax.set_title("%d walkers\nframe %d | max degree %d | avg degree %d)"%(self.numpoints,i,max(c),np.mean(c)))
            return self.scat,

        def show(self):
            plt.show()

class Swarm(list):
    record = False
    a = 1
    b = .2
    TimeStep = .05
    def  __init__(self, memory=None, *args, **kwargs):

        self.tree = None
        self.dt = 1
        self.G = Graph(memory)
        self.alpha = 100000
        self.p = Pool(10)

    def createNew(self, numNodes):
        self.unitConnectionRadius = 1. / numNodes**2
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
        #self.setPlotData()
        self.addForce()
        self.boundaryCondition()
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
        return np.sqrt(np.random.pareto(self.alpha, (self.len, 1)))

    def record(self):
        self.recordWalkers()

    def setPlotData(self):
        self.walkersPlot.set_data(self.getWalkersX(),self.getWalkersY())

    def getWalkersX(self):
        return self.walkers[:,0]

    def getWalkersY(self):
        return self.walkers[:,1]

    #def initRecord(self,*args,**kwargs):
    #    self.ani = AnimatedScatter(self.len,self.timeStep,(self.walkers,self.G.getConnectionNum()),kwargs)
        # self.fig = plt.figure()
        # self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        # self.ax = self.fig.add_subplot(111, aspect='equal', autoscale_on=False,
        #                                xlim = (0,1), ylim=(0, 1))
        # self.walkersPlot, = self.ax.plot(self.getWalkersX(), self.getWalkersY(), 'bo',ms=4)

    def recordWalkers(self, save = False, frames=6000, interval = 100):
        self.ani = AnimatedScatter(self.len, self.timeStep,
                                    self.getSwarm,save=save)

        #ani = animation.FuncAnimation(self.fig, self.timeStep, frames, interval=100)
        #if save: ani.save('%d-WithGraphR2.mp4' % (self.len), fps=60, extra_args=['-vcodec', 'libx264'])
        self.ani.show()

    def plotGraphConnections(self):
        plt.figure()
        counts = Counter(self.G.getConnectionNum())
        plt.bar(counts.keys(), counts.values())#self.G.getConnectionNum())
        plt.show()


if __name__ == "__main__":
    S = Swarm()
    S.createNew(100)
    S.recordWalkers()
    S.plotGraphConnections()
