from random import random, uniform
from math import pi, cos, sin, sqrt
import networkx as nx
from collections  import Counter

from operator import attrgetter
# -- t = 2*pi*random()
# -- u = random()+random()
# -- r = if u>1 then 2-u else u
# -- [r*cos(t), r*sin(t)]
import matplotlib.pyplot as plt
import matplotlib.animation as animation

WALKERS = 1000
unitConnectionRadius = sqrt(1./WALKERS)
class Walker:
    number=0
    def __init__(self):
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

walkers = [Walker() for i in range(WALKERS)]
fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-1, 1), ylim=(-1, 1))
walkersX = [walker.x for walker in walkers]
walkersY = [walker.y for walker in walkers]

walkersPlot, = ax.plot(walkersX, walkersY, 'bo',ms=2)

G = nx.Graph()

G.add_nodes_from(walkers)




def updateGraph():
    global G,walkers
    sortedWalkers = sorted(walkers,key=attrgetter('x','y'))
    print sortedWalkers


    for walker in walkers:
        neighbors = [w for w in walkers if w!=walker
                    and w.bumped(walker)]
        G.add_edges_from(zip([walker]*len(neighbors),neighbors))


def timeStep(i):
    global walkers,walkersPlot

    walkersPlot.set_data([walker.x for walker in walkers],
                         [walker.y for walker in walkers])
    map(lambda x: x.next(),walkers)
    updateGraph()
    return walkersPlot

# ani = animation.FuncAnimation(fig, timeStep, frames=60,
#                               interval=100)
# #ani.save('walkers.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
# plt.show()
for i in range(100):
    timeStep(i)

def plotGraphConnections(graph):
    counts = Counter(graph.degree().values())
    plt.figure()
    plt.bar(counts.keys(),counts.values())
    plt.show()

plotGraphConnections(G)
