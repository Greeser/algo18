import numpy as np


class Table:
    """ Hash table storage """
    def __init__(self):
        self.table = dict()

    def append_val(self, key, val):
        self.table.setdefault(key, []).append(val)

    def get_list(self, key):
        return self.table.get(key, [])

    def set_val(self, key, val):
        self.table[key] = val

    def get_val(self, key):
        return self.table[key]

    def keys(self):
        return self.table.keys()


class LSHashing(object):
    def __init__(self):
        self.name = 'LSHashing'

    def _init_uniform_planes(self):

        self.uniform_planes = [np.random.randn(self.size_of_hash, self.input_dim) for _ in range(self.n_hashtables)]

    def _init_hashtables(self):

        self.hash_tables = [Table() for _ in range(self.n_hashtables)]

    def _hash(self, planes, point):
        """ Calculate and return the binary hash for input_point and returns it. """

        point = np.array(point)
        projections = np.dot(planes, point)

        return "".join(['1' if i > 0 else '0' for i in projections])

    def add_point(self, input_point, index=None):
        """ Hash an input point into hash tables. """

        if index:
            value = (tuple(input_point), index)
        else:
            value = tuple(input_point)

        for i, table in enumerate(self.hash_tables):
            table.append_val(self._hash(self.uniform_planes[i], input_point),
                             value)

    def query(self, query_point, n=None):
        """
        Return all or n nearest neighbors.
        :param query_point: A list, or tuple, or numpy ndarray that only contains numbers.
        :param n: (optional) max amount of neighbours to be returned otherwise returns all.
        """

        neighbors = set()

        # add all near points from all tables to neighbors set
        for i, table in enumerate(self.hash_tables):
            binary_hash = self._hash(self.uniform_planes[i], query_point)
            neighbors.update(table.get_list(binary_hash))

        # rank neighbors based on distance
        neighbors = [(ix, LSHashing.euclidean_square(query_point, np.asarray(ix[0]))) for ix in neighbors]
        neighbors.sort(key=lambda x: x[1])

        return neighbors[:n] if n else neighbors

    def fit(self, X, hash_size=32, n_hashtables=10):
        """
        :param hash_size: (optional) The length of the resulting binary hash in integer.
        :param n_hashtables: (optional) The number of hash tables used for multiple lookups.
        """

        # determining number of dimensions
        self.input_dim = len(X[0])

        self.size_of_hash = hash_size
        self.n_hashtables = n_hashtables

        self._init_uniform_planes()
        self._init_hashtables()

        for i, x in enumerate(X):
            self.add_point(x, i)    # i - index - additional data that also put into table as value

    def predict(self, x, n=1):
        """
        :param x: query point
        :param n: (optional) number of nearest neighbors
        :return: list of indexes of nearest n points that correspond to the initial training set
        """
        result = self.query(x, n)
        return [point[0][1] for point in result]    # to get index from initial training set

    @staticmethod
    def euclidean(x, y):
        diff = np.array(x) - y
        return np.sqrt(np.dot(diff, diff))

    @staticmethod
    def euclidean_square(x, y):
        diff = np.array(x) - y
        return np.dot(diff, diff)


def main():
    lsh = LSHashing()
    x_train = [[1, 2, 3],
               [2, 3, 4],
               [1, 0, 3],
               [2, 0, 4],
               [2, 3, 0],
               [10, 12, 99]]

    lsh.fit(x_train, hash_size=12)
    print(lsh.predict([1, 0, 4], 2))


if __name__ == '__main__':
    main()
