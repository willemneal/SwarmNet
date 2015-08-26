
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


class AnimatedScatter(object):
        def __init__(self, numpoints, update, initState,save=False,fps=60):
            self.numpoints = numpoints
            self.update = update
            self.init = initState
            # Setup the figure and axes...
            self.fig, self.ax = plt.subplots(figsize=(10,10))
            self.ax.set_title("%d walkers"%(self.numpoints))
            # Then setup FuncAnimation.
            self.ani = animation.FuncAnimation(self.fig, self.next,frames=600, interval=100,
                                               init_func=self.setup_plot, blit=True)
            if save: self.ani.save('%d-Scatter2.mp4' % (self.numpoints), fps=fps, extra_args=['-vcodec', 'libx264'])

        def setup_plot(self):
            """Initial drawing of the scatter plot."""
            x, y, c = self.init()
            self.scat = self.ax.scatter(x, y, c=c, s=120, vmin=0,vmax=30,animated=True,cmap=plt.cm.get_cmap('winter'))
            #plt.colorbar(self.scat)
            self.ax.axis([-2, 2, -2, 2])
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
