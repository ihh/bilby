import pickle
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

filename = sys.argv[1]
with open (filename, mode="rb") as f:
    vars = pickle.load (f)
    f.close()

print(str(vars))
