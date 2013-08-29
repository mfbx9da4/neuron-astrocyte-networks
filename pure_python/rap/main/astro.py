from scipy import array, mean


class Astro(object):
    """docstring for SynapseAssociatedAstrocyte"""
    def __init__(self, syns, lag_iters):
        self.act = 0.
        self.syns = syns
        self._prev = [0. for i in range(lag_iters)]

    @property
    def prev(self):
        return float(mean(array(self._prev)))

    @prev.setter
    def prev(self, new):
        self._prev = self._prev[1:] + [new]


