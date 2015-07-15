class Memory(dict):
    def __init__(self,length):
        super(Memory,self).__init__()
        self.length = length


    def updateVals(self):
        self.toRemove = []
        for key in self.keys():

            self[tuple(key)] -= 1
            if self[tuple(key)] == 0: ##needs to be removed
                self.toRemove.append(key)
                del self[tuple(key)]

    def addExp(self, items):
        self.updateVals()
        self.update({tuple(key):self.length for key in items})
        return self.toRemove
