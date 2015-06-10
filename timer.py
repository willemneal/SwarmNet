import time

class Timer(object):
    def __init__(self, verbose=True):
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self
    @staticmethod
    def timeMe(f):
        def foo(*args,**kwargs):
            with Timer() as t:
                result =  f(args,kwargs)
            print "%s took %3d secs" % (f.func_name,t.secs)
            return result
        return foo

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            print 'elapsed time: %f ms' % self.msecs
