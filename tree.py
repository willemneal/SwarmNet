from sklearn.neighbors import BallTree as BT
import numpy as np
from scipy import spatial as sp


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
