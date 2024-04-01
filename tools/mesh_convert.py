import numpy as np
from sys import argv
import meshio
import h5py


def main(verbose=None):
    filename = argv[1]
    if verbose: 
        print(f'Reading mesh from {filename}', flush=True)
    mesh = meshio.read(filename)
    points = np.array(mesh.points, dtype=np.float64)
    elem = {'tet': [], 'prism': [], 'hex': []}
    for c in mesh.cells:
        key = c.type
        value = c.data
        if key == 'tetra':
            elem['tet'] = np.array(value, dtype=np.uint32)
        elif key == 'wedge':
            elem['prism'] = np.array(value, dtype=np.uint32)
        elif key == 'hexahedron':
            elem['hex'] = np.array(value, dtype=np.uint32)

    if verbose:
        print(f'Number of points: {len(points)}')
        print(f'Number of tetrahedra: {len(elem["tet"])}')
        print(f'Number of prisms: {len(elem["prism"])}')
        print(f'Number of hexahedra: {len(elem["hex"])}')
    # replace the extension with h5
    f_pc = filename.split('.')
    f_pc[-1] = 'h5'
    h5file = '.'.join(f_pc)
    if verbose:
        print(f'Writing mesh to {h5file}', flush=True)
    with h5py.File(h5file, 'w') as f:
        f.create_dataset(f'mesh/xg', data=points.flatten())
        for key, value in elem.items():
            if len(value) > 0:
                f.create_dataset(f'mesh/ien/{key}', data=value.flatten())
    if verbose:
        print('Converted mesh to h5 format', flush=True)
    # write a vtu
    f_pc[-1] = 'vtu'
    vtufile = '.'.join(f_pc)
    if verbose:
        print(f'Writing mesh to {vtufile}', flush=True)
    meshio.write(vtufile, mesh)

if __name__ == '__main__':
    main(verbose=True)

