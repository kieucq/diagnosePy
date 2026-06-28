import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import modColorCode as color

def metadataWRF(df):
    coords = list(df.coords)
    vars = list(df.data_vars)
    dims = dict(df.sizes) #dict(df.dims)
    print(f"List of coordinate names is: {coords}")
    print(f"List of vars is {vars}")
    print(f"List of dim names is {dims}")

def windAgrid3D(u,v,nx,ny):
    u1 = np.zeros((u.shape[0],ny,nx))
    v1 = np.zeros((u.shape[0],ny,nx))
    s1 = np.zeros((u.shape[0],ny,nx))
    u1[:,:,:] = 0.5*(u[:,:,0:nx]+u[:,:,1:nx+1])
    v1[:,:,:] = 0.5*(v[:,0:ny,:]+v[:,1:ny+1,:])
    s1 = np.sqrt(np.square(u1)+np.square(v1)) 
    #for j in range(ny):
    #    for i in range(nx):
    #        for k in range(u.shape[1]):
    #            s1[k,j,i] = np.sqrt(u1[k,j,i]**2 + v1[k,j,i]**2)
    return u1,v1,s1

def minPressureIndex(ps):
    min_index = np.unravel_index(np.argmin(ps), ps.shape)
    print(f"min index is {min_index}")
    return min_index

def domainSize(ps,nx,ny,radius,dx):
    print(f"Input surface pressure shape is {ps.shape}")
    rcen,ccen = minPressureIndex(ps)
    print(f"Storm center is at {rcen,ccen}")
    cs = max(1,ccen - round(radius/dx))
    ce = min(ccen + round(radius/dx),nx)
    rs = max(1,rcen - round(radius/dx))
    re = min(rcen + round(radius/dx),ny)
    return cs,ce,rs,re,rcen,ccen

def loadSingleWRF(infile):
    # open file now and print out meta data for quick look
    df = xr.load_dataset(infile)
    metadataWRF(df)

    # pull information about dim for A-grid plot
    tt = df['T'].values
    t1 = tt[0]
    nx = tt.shape[3]
    ny = tt.shape[2]
    nz = tt.shape[1]
    print(f"A-grid dimension for ploting (nx,ny,nz) are {nx,ny,nz}")

    # extract u,v componet for 1 level. The row can be reversed by using
    # a change in index u1000r = uu[0,0,::-1, :]
    uu = df['U'].values
    vv = df['V'].values
    u1,v1,s1 = windAgrid3D(uu[0],vv[0],nx,ny)
    print(f"wind speed shape is {s1.shape}")

    # extract vertical motion and tunr to A-grid
    ww = df['W'].values
    w1 = 0.5*(ww[0,0:nz,:,:]+ww[0,1:nz+1,:,:])

    qq = df['QVAPOR'].values
    q1 = qq[0]
    ps = df['PSFC'].values
    ps1 = ps[0] 
    return u1,v1,w1,s1,t1,q1,ps1,nx,ny,nz
    
def plotSpeedUVfull(u1,v1,s1,ilev,cs,ce,rs,re,density,dx,nx,ny):
    # Create contour plot
    plt.figure(figsize=(7, 5))
    Lx = (nx-1)*dx
    Ly = (ny-1)*dx
    X = np.linspace(-Lx/2, Lx/2, nx)   # X-axis values
    Y = np.linspace(-Ly/2, Ly/2, ny)   # Y-axis values
    x, y = np.meshgrid(X, Y)

    # plot color shaded using my contour levels
    clevs = [1.,3.,5.,7.,9.,12.,15.,18.,21.,25.,30.,35.,40.,100.]
    bcols = color.myColor(clevs,(0.9,0.9,1.),(0.,0.,1.))
    plt.contourf(x[rs:re,cs:ce], y[rs:re,cs:ce], s1[ilev,rs:re,cs:ce], levels=clevs, colors=bcols)

    # Add a custom colorbar
    plt.colorbar(label="Wind (m/s)",ticks=[5,10,15,20,25,30,35,40])

    # add vector
    plt.quiver(x[rs:re:density, cs:ce:density], y[rs:re:density, cs:ce:density], 
               u1[ilev,rs:re:density, cs:ce:density], v1[ilev,rs:re:density, cs:ce:density], 
               scale=300,color='black')

    # title and axis label
    plt.title("WRF output from 2km resolution")
    plt.xlabel("X (km)")
    plt.ylabel("Y (km)")

    plt.show()

def plotSpeedUVzoom(u1,v1,s1,ilev,cs,ce,rs,re,density,dx,nx,ny):
    # Create contour plot
    plt.figure(figsize=(7, 5))
    Lx = (ce-cs+1)*dx
    Ly = (re-rs+1)*dx
    X = np.linspace(-Lx/2, Lx/2, ce-cs)   # X-axis values
    Y = np.linspace(-Ly/2, Ly/2, re-rs)   # Y-axis values
    x, y = np.meshgrid(X, Y)

    # plot color shaded using my contour levels
    clevs = [1.,3.,5.,7.,9.,12.,15.,18.,21.,25.,30.,35.,40.,100.]
    bcols = color.myColor(clevs,(0.9,0.9,1.),(0.,0.,1.))
    plt.contourf(x, y, s1[ilev,rs:re,cs:ce], levels=clevs, colors=bcols)

    # Add a custom colorbar
    plt.colorbar(label="Wind (m/s)",ticks=[5,10,15,20,25,30,35,40])

    # add vector
    plt.quiver(x[::density, ::density], y[::density, ::density],
               u1[ilev,rs:re:density, cs:ce:density], v1[ilev,rs:re:density, cs:ce:density],
               scale=300,color='black')

    # title and axis label
    plt.title("WRF output from 2km resolution")
    plt.xlabel("X (km)")
    plt.ylabel("Y (km)")
    plt.xticks(np.arange(-180, 181, 60))
    plt.yticks(np.arange(-180, 181, 60))

    plt.show()

def plotSpeedUVWfull(u,v,w,s,ilev,cs,ce,rs,re,density,dx,nx,ny):
    # Create contour plot
    plt.figure(figsize=(7, 5))
    Lx = (nx-1)*dx
    Ly = (ny-1)*dx
    X = np.linspace(-Lx/2, Lx/2, nx)   # X-axis values
    Y = np.linspace(-Ly/2, Ly/2, ny)   # Y-axis values
    x, y = np.meshgrid(X, Y)

    # plot color shaded using my contour levels
    clevs = [-5, -4, -3, -2, -1, -0.5, -0.01, 0.01, 0.5, 1, 2, 3, 4, 5]
    bcols = color.myColor(clevs[0:7],(0,0,1),(1.,1.,1.))
    rcols = color.myColor(clevs[6:14],(1,1,1),(1.,0.,0.))
    brcols = bcols + rcols
    plt.contourf(x[rs:re,cs:ce], y[rs:re,cs:ce], w[ilev,rs:re,cs:ce]*10, levels=clevs, colors=brcols)

    # Add a custom colorbar
    plt.colorbar(label="Wind (cm/s)",ticks=[-5, -4, -3, -2, -1, -0.5, -0.01, 0.01, 0.5, 1, 2, 3, 4, 5])

    # add vector
    plt.quiver(x[rs:re:density, cs:ce:density], y[rs:re:density, cs:ce:density],
               u[ilev,rs:re:density, cs:ce:density], v[ilev,rs:re:density, cs:ce:density],
               scale=300,color='black')

    # add contour w
    plt.contour(x[rs:re,cs:ce], y[rs:re,cs:ce], s[ilev,rs:re,cs:ce], colors='k')

    # title and axis label
    plt.title("WRF output from 2km resolution")
    plt.xlabel("X (km)")
    plt.ylabel("Y (km)")

    plt.show()

def plotSpeedUVWzoom(u,v,w,s,ilev,cs,ce,rs,re,density,dx,nx,ny,outfile):
    # Create contour plot
    plt.figure(figsize=(7, 5))
    Lx = (ce-cs-1)*dx
    Ly = (re-rs-1)*dx
    X = np.linspace(-Lx/2, Lx/2, ce-cs)   # X-axis values
    Y = np.linspace(-Ly/2, Ly/2, re-rs)   # Y-axis values
    x, y = np.meshgrid(X, Y)

    # plot color shaded using my contour levels
    blevs = [-70, -50, -30, -10, -5, -3, -1, -0.5, 0]
    rlevs = [0, 0.5, 1, 3, 5, 10, 30, 50, 70]
    clevs = blevs + rlevs[1:]
    bcols = color.myColor(blevs,(0,0,1),(1.,1.,1.))
    rcols = color.myColor(rlevs,(1,1,1),(1.,0.,0.))
    brcols = bcols + rcols
    plt.contourf(x, y, w[ilev,rs:re,cs:ce]*10, levels=clevs, colors=brcols)
    #plt.contourf(x, y, w[ilev,rs:re,cs:ce]*10, levels=clevs,cmap='bwr')

    # Add a custom colorbar
    plt.colorbar(label="(cm/s)",ticks=[-50, -30, -10, -5, -3, -1, -0.5, 0.5, 1, 3, 5, 10, 30, 50])

    # add vector
    plt.quiver(x[::density, ::density], y[::density, ::density],
               u[ilev,rs:re:density, cs:ce:density], v[ilev,rs:re:density, cs:ce:density],
               scale=800,color='black')

    # add contour w
    plt.contour(x, y, s[ilev,rs:re,cs:ce], colors='k')

    # title and axis label
    plt.title(f"WRF output from {dx}-km resolution")
    plt.xlabel("X (km)")
    plt.ylabel("Y (km)")
    plt.xticks(np.arange(-120, 121, 30))
    plt.yticks(np.arange(-120, 121, 30))
    plt.savefig(outfile)

    plt.show()

#================================================================================================
#
# This section is to directly test this module as a stand-alone code
#
if  __name__ == "__main__":

    # loading WRF input data
    infile = "wrfout_d01.new"
    outfile = "fig_d01.png"
    u,v,w,spd,t,qv,ps,nx,ny,nz = loadSingleWRF(infile)

    # set a level and domain to plot
    dx = 36
    ilev = 1
    radius = 120
    density = 1
    cs,ce,rs,re,rcen,ccen = domainSize(ps,nx,ny,radius,dx)
    print(f"Domain center and size to plot are {rcen,ccen,cs,ce,rs,re}")

    # plot wind speed and vector
    #plotSpeedUVzoom(u,v,spd,ilev,cs,ce,rs,re,density,dx,nx,ny)
    
    # plot wind speed and vector and w contour
    plotSpeedUVWzoom(u,v,w,spd,ilev,cs,ce,rs,re,density,dx,nx,ny,outfile)
