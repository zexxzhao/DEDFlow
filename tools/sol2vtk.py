import numpy as np
import meshio
import h5py

def get_options():
    '''
    Get options:
    argv[1]: mesh file eg. mesh.h5
    argv[2]: solution file eg. sol.%d.h5
    if len(argv) == 4:
        argv[3]: this is the step number
    elif len(argv) == 6:
        argv[3]: start step number
        argv[4]: end step number
        argv[5]: step size
    
    '''
    import argparse
    parser = argparse.ArgumentParser(description='Convert restart files to XDMF format')
    parser.add_argument('m', help='Mesh file')
    parser.add_argument('i', help='Solution file')
    parser.add_argument('args', nargs='*', help='Arguments')
    return parser.parse_args()


def read_mesh(mesh_file):
    '''
    Read mesh file
    '''
    import h5py
    with h5py.File(mesh_file, 'r') as f:
        xg = np.asarray(f['mesh/xg']).reshape(-1, 3)
        try:
            etet = np.asarray(f['mesh/ien/tet']).reshape(-1, 4)
        except KeyError:
            etet = None
        try:
            eprism = np.asarray(f['mesh/ien/prism']).reshape(-1, 6)
        except KeyError:
            eprism = None
        try:
            ehex = np.asarray(f['mesh/ien/hex']).reshape(-1, 8)
        except KeyError:
            ehex = None

    return xg, etet, eprism, ehex


def read_sol(sol_file):
    '''
    Read sol file
    '''
    sol = {}
    print('\tDataset:', end='')
    with h5py.File(sol_file, 'r') as f:
        for key in f.keys():
            print(f' [{key}]', end='')
            sol[key] = np.asarray(f[key])
            print(f' {sol[key].shape}', end='')
    return sol


def write_xdmf(filename, mesh, point_data, cell_data=None):
    '''
    Write XDMF file
    '''
    xg, etet, eprism, ehex = mesh
    nnode = xg.shape[0]
    for key in point_data.keys():
        point_data[key] = point_data[key].reshape(nnode, -1).squeeze()

    if cell_data is not None:
        nelem = etet.shape[0] + eprism.shape[0] + ehex.shape[0]
        for key in cell_data.keys():
            cell_data[key] = cell_data[key].reshape(nelem, -1).squeeze()

    # Write mesh
    m_points = xg
    m_cells = {}
    if etet is not None:
        m_cells['tetra'] = etet
    if eprism is not None:
        m_cells['wedge'] = eprism
    if ehex is not None:
        m_cells['hexahedron'] = ehex
    m_point_data = point_data
    m_cell_data = cell_data

    meshio.write_points_cells(filename, m_points, m_cells,
                              point_data=m_point_data, cell_data=m_cell_data)
    # meshio.write(filename, meshio.Mesh(points=m_points, cells=m_cells, point_data=m_point_data, cell_data=m_cell_data))


def main():
    '''
    Main function
    '''
    # Get options
    args = get_options()
    mesh_file = args.m
    sol_file = args.i
    if len(args.args) == 1:
        start = int(args.args[0])
        end = start
        step = 1
    elif len(args.args) == 3:
        start, end, step = map(int, args.args)
    else:
        raise ValueError('Invalid number of arguments')

    print('mesh_file:', mesh_file)
    print('sol_file:', sol_file)
    print('[start, end, step]:', [start, end, step])

    # Read mesh
    mesh = read_mesh(mesh_file)

    # Read solution
    keys = ['u', 'v', 'w', 'p', 'T', 'rho', 'mu', 'k', 'e', 's', 'vorticity']
    for i in range(start, end+1, step):
        sol_file_i = sol_file % i
        print('Reading:', sol_file_i, end='', flush=True)
        sol = read_sol(sol_file_i)

        # Write XDMF file
        filename = 'w.%d.xdmf' % i
        print('\tWriting:', filename, flush=True)
        write_xdmf(filename, mesh, sol, None)


if __name__ == '__main__':
    main()
