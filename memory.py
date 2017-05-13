import numpy as np

class Memory(object):
    def __init__(self, dimension=10, size=10000):
        self.memory = np.empty((size,dimension), dtype=np.float32)
        self.size = size
        self.index = 0 # keeps track of current size
        self.full = False
    def add(self, memory):
        self.memory[self.index,:] = memory
        self.index += 1
        if self.index >= self.size:
            self.index = 0
            self.full = True
    def sample(self, n):
        # data stored in columns
        # each column is one entry
        if self.full:
            idx = np.random.randint(self.size, size=n)
        else:
            idx = np.random.randint(self.index, size=n)
        return self.memory[idx,:]

class MultiMemory(object): # assumes that all data can at least be represented by floats
    def __init__(self, size=1000):
        self.memory = {}
        self.size = size
        self.index = 0
        self.full = False

    def add(self, mem_dict):
        for k,v in mem_dict.iteritems():
            v = np.asarray(v)

            if not self.memory.has_key(k):
                vs = (1,) if (len(v.shape) == 0) else v.shape
                s = (self.size,) + vs
                self.memory[k] = np.empty(s, dtype=v.dtype)

            self.memory[k][self.index,:] = v # TODO : copy?

        self.index += 1
        if self.index >= self.size:
            self.index = 0
            self.full = True

    def sample(self, n):
        if self.full:
            idx = np.random.randint(self.size, size=n)
        else:
            idx = np.random.randint(self.index, size=n)
        res = {}
        for k,v in self.memory.iteritems():
            res[k] = v[idx,:]
        return res

if __name__ == "__main__":
    m = MultiMemory(size = 100)

    for i in range(200):
        s = np.random.random((5,5))
        a = np.random.randint(10)
        m.add({'s': s, 'a' : a})

    print m.sample(10)['s'].shape
