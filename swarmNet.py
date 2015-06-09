from random import random
from math import pi, cos, sin
# -- t = 2*pi*random()
# -- u = random()+random()
# -- r = if u>1 then 2-u else u
# -- [r*cos(t), r*sin(t)]
import matplotlib.pyplot as plt

class Walker:
    def __init__(self):
        u = random()+random()
        self.r = u if u<=1 else 2-u
        self.t = 2*pi*random()
        self.x =  self.r*cos(self.t)
        self.y =  self.r*sin(self.t)

walkers = [Walker() for i in range(1000)]

def plotWalkers():
    plt.scatter([walker.x for walker in walkers],
            [walker.y for walker in walkers])

    plt.show()

plotWalkers()
