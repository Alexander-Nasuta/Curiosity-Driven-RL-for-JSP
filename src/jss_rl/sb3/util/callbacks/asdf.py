import numpy
import numpy as np

if __name__ == '__main__':
    mem = np.empty(shape=(0, 2))

    mem = np.vstack((mem, numpy.array([0., 1.])))
    mem = np.vstack((mem, numpy.array([0., 2.])))
    mem = np.vstack((mem, numpy.array([0., 3.])))
    mem = np.vstack((mem, numpy.array([0., 13.])))

    print(np.mean(mem, axis=0))
    print(np.max(mem, axis=0))
    print(np.min(mem, axis=0))


    res = np.mean(mem[-100:],  axis=0)
    print(res)
