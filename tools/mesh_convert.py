import warnings
import numpy as np
from sys import argv
import meshio
import h5py


keys = ['tetra', 'wedge', 'hexahedron', 'triangle']
key_map = {'tetra': 'tet', 'wedge': 'prism', 'hexahedron': 'hex', 'triangle': 'tri'} 

def get_mesh_coord(mesh, verbose=False):
    xg =np.asarray(mesh.points, dtype=np.float64)
    if verbose:
        print(f'Number of points: {len(xg)}')
    return xg


def get_mesh_elem(mesh, verbose=False):
    elem = {}

    for c in mesh.cells:
        k, v = c.type, c.data
        if k in keys:
            elem[key_map[k]] = np.array(v, dtype=np.uint32)
        else:
            warnings.warn(f'Unknown cell type: {k}')
    if verbose:
        for k, v in elem.items():
            print(f'Number of {k}: {len(v)}')
    return elem


def get_mesh_facet_connectivity(elem):
    tet = elem.get('tet', []).reshape(-1, 4)
    tri = elem.get('tri', []).reshape(-1, 3)

    # build a v2e map
    v2e = {}
    for i, e in enumerate(tet):
        for v in e:
            if v in v2e:
                v2e[v].append(i)
            else:
                v2e[v] = [i]
    for k, v in v2e.items():
        v2e[k] = list(set(v))

    # build a f2e map
    f2e = np.zeros(len(tri), dtype=np.uint32)
    for i, f in enumerate(tri):
        all_e = []
        for v in f:
            all_e.extend(v2e[v])
        # find the most appeared element
        f2e[i] = max(set(all_e), key=all_e.count)

    # build facet orientation
    forn = np.zeros(len(tri), dtype=np.uint32)
    for i, f in enumerate(tri):
        e = f2e[i]
        e_v = tet[e]
        # find the local index of the face
        local_idx = np.argwhere(np.logical_not(np.isin(e_v, f))).squeeze()
        forn[i] = local_idx

    return f2e, forn


def main(verbose=None):
    filename = argv[1]
    if verbose: 
        print(f'Reading mesh from {filename}', flush=True)
    mesh = meshio.read(filename)
    points = np.array(mesh.points, dtype=np.float64)
    # elem = {'tet': [], 'prism': [], 'hex': []}
    elem = get_mesh_elem(mesh, verbose)
    cell_data_key = 'gmsh:physical'
    cell_data = {}
    cell_data['tri'] = mesh.cell_data[cell_data_key][0]
    cell_data['tet'] = mesh.cell_data[cell_data_key][1]

    # sort facet using cell_data as the key
    facet = np.arange(len(elem['tri']))
    sorted_facet = sorted(facet, key=lambda x: cell_data['tri'][x])
    elem['tri'] = elem['tri'][sorted_facet]

    _, count = np.unique(cell_data['tri'], return_counts=True)
    elem_offset = np.cumsum(count)
    elem_offset = np.insert(elem_offset, 0, 0)

    # generate the connectivity between facets and elements
    f2e, forn = get_mesh_facet_connectivity(elem)

    # generate bnode for facets
    bnode = []
    bnode_count = []
    for i in range(len(elem_offset)-1):
        start, end = elem_offset[i], elem_offset[i+1]
        facets = elem['tri'][start:end].flatten()
        facets = np.unique(facets)
        bnode_count.append(len(facets))
        bnode.extend(facets)

    bnode = np.array(bnode, dtype=np.uint32)
    bnode_offset = np.cumsum(bnode_count)
    bnode_offset = np.insert(bnode_offset, 0, 0)

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
        f.create_dataset(f'mesh/bound/node_offset', data=bnode_offset)
        f.create_dataset(f'mesh/bound/node', data=bnode)
        f.create_dataset(f'mesh/bound/elem_offset', data=elem_offset)
        f.create_dataset(f'mesh/bound/ien', data=elem['tet'].flatten())
        f.create_dataset(f'mesh/bound/f2e', data=f2e)
        f.create_dataset(f'mesh/bound/forn', data=forn)
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

