import sys
import h5py as h5
import numpy as np

if len(sys.argv) != 2 and len(sys.argv) != 3:
    print('Usage: {} file.h5 [key]'.format(sys.argv[0]))
    sys.exit(1)

def scan_hdf5(path, recursive=True, tab_step=2):
    def scan_node(g, tabs=0):
        elems = []
        for k, v in g.items():
            if isinstance(v, h5.Dataset):
                elems.append(v.name)
            elif isinstance(v, h5.Group) and recursive:
                elems.append((v.name, scan_node(v, tabs=tabs + tab_step)))
        return elems
    with h5.File(path, 'r') as f:
        return scan_node(f)

filename = sys.argv[1]
if len(sys.argv) == 3:
    key = sys.argv[2]

    f = h5.File(filename, 'r')
    val = f.get(key)

    if val is None:
        print('Key not found')
        sys.exit(1)

    print(np.array(val))

else:
    print(scan_hdf5(filename))
