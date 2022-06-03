import numpy as np

class MovingAverage:
    """Computes the moving average of a variable."""

    def __init__(self, capacity):
        self._capacity = capacity
        self._history = np.array([0.0] * capacity)
        self._size = 0

    def add(self, value):
        index = self._size % self._capacity
        self._history[index] = value
        self._size += 1

    def mean(self):
        if not self._size:
            return None
        if self._size < self._capacity:
            return np.mean(self._history[0:self._size])
        return np.mean(self._history)