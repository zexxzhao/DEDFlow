import numpy as np
import vtk

def get_options():
    '''
    Get options from command line
    -m : [*.h5] mesh file name
    -i : [sol.%d.h5] restart file name
    *args : if len(args) == 1: step = int(args[0])
           if len(args) == 3: start, end, step = map(int, args)
    '''
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str, required=True)
    parser.add_argument('-i', type=str, required=True)
    parser.add_argument('args', nargs='*')
    return parser.parse_args()

def read_mesh(mesh_file):
    '''
    Read mesh file
    '''
    import h5py
    with h5py.File(mesh_file, 'r') as f:
        xg = f['mesh/xg'].reshape(-1, 3)
        etet = f['mesh/ien/tet'].reshape(-1, 4)
        eprism = f['mesh/ien/prism'].reshape(-1, 6)
        ehex = f['mesh/ien/hex'].reshape(-1, 8)

    return xg, etet, eprism, ehex

def read_restart(restart_file):
    '''
    Read restart file
    '''
    import h5py
    with h5py.File(restart_file, 'r') as f:
        step = f.attrs['step']
        time = f.attrs['time']
        u = f['u'].reshape(-1, 3)
        v = f['v'].reshape(-1, 3)
        p = f['p'].reshape(-1, 1)

    return step, time, u, v, p

