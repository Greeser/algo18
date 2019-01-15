import h5py

def get_mnist(fn):
    """
    fn: path-to-file
    """
    try:
        f = h5py.File(fn, 'r')
        train, test, neighbor = f['train'], f['test'], f['neighbors']
        return train, test, neighbor
    except:
        print("MNIST dataset not found. Location:" + fn)
        
def get_sift(fn):
    """
    fn: path-to-file
    """
    try:
        f = h5py.File(fn, 'r')
        train, test, neighbor = f['train'], f['test'], f['neighbors']
        return train, test, neighbor
    except:
        print("SIFT dataset not found. Location:" + fn)

