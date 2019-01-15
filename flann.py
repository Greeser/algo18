import pyflann

class FLANN(object):
    def __init__(self):
        self.name = "FLANN"
        
    def fit(self, X, **kwargs):
        """
        target_precision = float [0,1]
        algorithm = [linear, kdtree, kmeans, composite, autotuned]
        log_level = info
        """
        self._flann = pyflann.FLANN(**kwargs)
        self._flann.build_index(X)

    def predict(self, X, n):
        return self._flann.nn_index(X,n)[0][0]
