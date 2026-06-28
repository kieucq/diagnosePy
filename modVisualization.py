import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

class array2D:
    def __init__(self,nx,ny):
        self.nx = nx
        self.ny = ny

class array3D:
    def __init__(self,nx,ny,nz):
        self.nx = nx
        self.ny = ny
        self.nz = nz

class array4D:
    def __init__(self,nx,ny,nz,nt):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.nt = nt

