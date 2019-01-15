import h5py

f = h5py.File("mnist-784-euclidean.hdf5", 'r')

data = f['train']

for key in f.keys():
    print(key)

nb = f['neighbors']

print(nb.shape)

test = f['test']

print(test.shape)
