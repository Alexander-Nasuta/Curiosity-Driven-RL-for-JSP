import numpy as np

from collections import deque

if __name__ == '__main__':
    deq = deque(maxlen=4)
    deq.append("a")
    print(deq)
    deq.append("b")
    deq.append("c")
    print(deq)
    deq.append("d")
    deq.append("e")
    print(deq)

    res = np.array(deq)
    print(res)


