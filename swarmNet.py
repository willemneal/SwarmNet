from random import random, uniform
from math import pi, cos, sin, sqrt
import networkx as nx
# -- t = 2*pi*random()
# -- u = random()+random()
# -- r = if u>1 then 2-u else u
# -- [r*cos(t), r*sin(t)]
import matplotlib.pyplot as plt
import matplotlib.animation as animation

WALKERS = 100

class Walker:
    def __init__(self):
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


walkers = [Walker() for i in range(WALKERS)]
fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-1, 1), ylim=(-1, 1))
walkersX = [walker.x for walker in walkers]
walkersY = [walker.y for walker in walkers]

walkersPlot, = ax.plot(walkersX, walkersY, 'bo',ms=6)

G = nx.Graph()

G.add_nodes_from(walkers)

unitConnectionRadius = sqrt(1./len(walkers))


def updateGraph(g):
    pass


def animate(i):
    global walkers,walkersPlot

    walkersPlot.set_data([walker.x for walker in walkers],
                         [walker.y for walker in walkers])
    map(lambda x: x.next(),walkers)
    return walkersPlot

ani = animation.FuncAnimation(fig, animate, frames=6000,
                              interval=100)
#ani.save('walkers.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
plt.show()
