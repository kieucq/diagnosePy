import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import modColorCode as color
from matplotlib import colors
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import ListedColormap

# load wrfinput data
def loadWRF(infile):
    # open file now and print out metadata for quick look
    df = xr.load_dataset(infile)

    # pull information about dim for A-grid plot
    tt = df['T'].values
    t1 = tt[0]
    nx = tt.shape[3]
    ny = tt.shape[2]
    nz = tt.shape[1]
    dx = df.DX
    print(f"A-grid dim (dx,nx,ny,nz) for {infile} are {dx,nx,ny,nz}")

    return df,dx,nx,ny,nz

#list all info
def infoAll(df):
    print(f"List of coordinate names:")
    for coord_name, coord_data in df.coords.items():
        print(f"Coordinate name is: {coord_name}")

    print(f"List of vars is")
    for var_name, var_data in df.data_vars.items():
        print(f"Variable names is: {var_name}")

    print(f"List of dim names is")
    for dim_name, dim_data in df.sizes.items():
        print(f"Dimension name is: {dim_name}, {dim_data}")

# list all 3D vars
def info3D(df):
    for var_name, var_data in df.data_vars.items():
        if len(var_data.dims) == 4:
            print(f"Variable: {var_name}, {var_data.dims}")

# list all 2D vars
def info2D(df):
    for var_name, var_data in df.data_vars.items():
        if len(var_data.dims) == 3:
            print(f"Variable: {var_name}, {var_data.dims}")

# find lat/lon index of a plotting domain
def lat_lon_id(df,slat,elat,slon,elon):
    lat1d = df.XLAT.values[0,:,0]
    lon1d = df.XLONG.values[0,0,:]
    rs = (np.abs(slat - lat1d)).argmin()
    re = (np.abs(elat - lat1d)).argmin()
    cs = (np.abs(slon - lon1d)).argmin()
    ce = (np.abs(elon - lon1d)).argmin()
    return rs,re,cs,ce

# function to compute 3D wind speed from (u,v) component
def wind_speed3D(u,v,nx,ny):
    u1 = np.zeros((u.shape[0],ny,nx))
    v1 = np.zeros((u.shape[0],ny,nx))
    s1 = np.zeros((u.shape[0],ny,nx))
    u1[:,:,:] = 0.5*(u[:,:,0:nx]+u[:,:,1:nx+1])
    v1[:,:,:] = 0.5*(v[:,0:ny,:]+v[:,1:ny+1,:])
    s1 = np.sqrt(np.square(u1)+np.square(v1)) 
    return u1,v1,s1

# function to find the index of the min center pressure on (x,y) plane
def min_pressure_center_index(ps):
    min_index = np.unravel_index(np.argmin(ps), ps.shape)
    print(f"min index is {min_index}")
    return min_index

# function to define a domain that follows a storm center with a given radius/dx
def storm_following_domain_center(ps,nx,ny,radius,dx):
    print(f"Input surface pressure shape is {ps.shape}")
    rcen,ccen = minPressureIndex(ps)
    print(f"Storm center is at {rcen,ccen}")
    cs = max(1,ccen - round(radius/dx))
    ce = min(ccen + round(radius/dx),nx)
    rs = max(1,rcen - round(radius/dx))
    re = min(rcen + round(radius/dx),ny)
    return cs,ce,rs,re,rcen,ccen

# class of 3D objects to be plotted
class var3D:
    # passing a 3D dimension to initialize 
    def __init__(self,nx=1,ny=1,nz=1,dx=1):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.nx1 = nx+1
        self.ny1 = ny+1
        self.nz1 = nz+1
        self.dx = dx

    # re-stagging U-wind component
    def shiftU3D(self,u):
        u1 = np.zeros((u.shape[1],self.ny,self.nx))
        u1[:,:,:] = 0.5*(u[0,:,:,0:self.nx]+u[0,:,:,1:self.nx1])
        return u1
 
    # re-stagging U-wind component
    def shiftV3D(self,v):
        v1 = np.zeros((v.shape[1],self.ny,self.nx))
        v1[:,:,:] = 0.5*(v[0,:,0:self.ny,:]+v[0,:,1:self.ny1,:])
        return v1 

    # plot a shading map at one given z-level on the gridded map (in km for x and y axes)
    def plot_zlevel_grid3D(self,a,cvar=None,
                           u3d=[],v3d=[],
                           ilev=1,rs=1,re=2,cs=1,ce=2,
                           clevs=[-999],title="WRF",
                           minc=(0,0,0),
                           maxc=(0,0,1),
                           labels="Ticks labels",
                           ticks=[-1,0,1],
                           ccolor="black",cthick=1.0,                          
                           scale=500,skipx=1,skipy=1,
                           option="shading",cfont=12,
                           vector=False,maplinewidth=0.5):

        # Create contour plot
        plt.figure(figsize=(7, 5))
        Lx = (self.nx-1)*self.dx
        Ly = (self.ny-1)*self.dx
        X = np.linspace(-Lx/2, Lx/2, self.nx)   # X-axis values
        Y = np.linspace(-Ly/2, Ly/2, self.ny)   # Y-axis values
        x, y = np.meshgrid(X, Y)

        # create my own color table
        bcols = color.myColor(clevs,minc,maxc)
        if clevs == [-999]:
            print("Plotting default color range")
            my_cmap = plt.cm.coolwarm
        else:
            print("Plotting customized color range")
            my_cmap, norm = color.make_cmap(clevs,neg_color=minc, pos_color=maxc)

        # plot either contour or shading mode
        if option == "shading":
            if clevs == [-999]:
                plot_out = plt.contourf(x[rs:re,cs:ce], y[rs:re,cs:ce], a[ilev,rs:re,cs:ce], cmap=my_cmap)
                if cvar.any() != None:
                    plot_contour = plt.contour(x[rs:re,cs:ce], y[rs:re,cs:ce], cvar[ilev,rs:re,cs:ce], colors=ccolor, linewidths=cthick)
            else:
                plot_out = plt.contourf(x[rs:re,cs:ce], y[rs:re,cs:ce], a[ilev,rs:re,cs:ce],levels=clevs, colors=bcols)
                if cvar.any() != None:
                    plot_contour = plt.contour(x[rs:re,cs:ce], y[rs:re,cs:ce], cvar[ilev,rs:re,cs:ce], 
                                               levels=clevs, colors=ccolor, linewidths=cthick)
            cbar = plt.colorbar(plot_out, location='bottom', pad=0.05)
            cbar.set_label(labels)
        else:
            if clevs == [-999]:
                plot_out = plt.contour(x[rs:re,cs:ce], y[rs:re,cs:ce], a[ilev,rs:re,cs:ce], cmap=my_cmap)
            else:
                plot_out = plt.contour(x[rs:re,cs:ce], y[rs:re,cs:ce], a[ilev,rs:re,cs:ce], cmap=my_cmap, levels=clevs)
            plt.clabel(plot_out, inline=True, fontsize=cfont, fmt='%d')

        # adding a vector on top
        if vector:
            u1,v1 = u3d[ilev,rs:re,cs:ce],v3d[ilev,rs:re,cs:ce]
            skip = (slice(None, None, skipy), slice(None, None, skipx))  # skip 2 arrows in row, and 3 arrow in cols dim

            Q = plt.quiver(x[skip], y[skip], u1[skip], v1[skip], scale=scale, color='k')
            plt.quiverkey(Q, 0.9, -0.2, 10, '10 m/s', labelpos='E')

        # title and axis label
        plt.title(f"{title} at level k = {ilev}")
        plt.xlabel("X (km)")
        plt.ylabel("Y (km)")
        plt.show()

    # plot a contour or shading map at one given z-level on the world map with wind vector field
    def plot_zlevel_world3D(self,df,a,cvar=None,
                           u3d=[],v3d=[],ilev=1,
                           slat=-90,elat=90,
                           slon=-180,elon=180,
                           clevs=[-999],
                           title="WRF",
                           minc=(0,0,0),
                           maxc=(1,1,1),
                           ccolor="black",cthick=1.0,
                           labels="Ticks labels",
                           ticks=[-1,0,1],
                           scale=500,skipx=1,skipy=1,
                           option="shading",cfont=12,
                           vector=False,maplinewidth=0.5):

        # locate the lat and lon index
        rs,re,cs,ce = lat_lon_id(df,slat,elat,slon,elon)
        NX = min(max(ce-cs,5),a.shape[1])
        NY = min(max(re-rs,5),a.shape[0])

        # re-define lat/lon grid start and end points
        slon_new, elon_new = df.XLONG[0,0,cs], df.XLONG[0,0,ce]
        slat_new, elat_new = df.XLAT[0,rs,0], df.XLAT[0,re,0]

        # Create the Basemap
        fig, ax = plt.subplots(figsize=(12, 6))
        m = Basemap(projection='merc',
                    llcrnrlat=slat_new, urcrnrlat=elat_new,
                    llcrnrlon=slon_new, urcrnrlon=elon_new,
                    resolution='l', ax=ax)

        # create my own color table
        bcols = color.myColor(clevs,minc,maxc)
        if clevs == [-999]:
            print("Plotting default color range")
            my_cmap = plt.cm.coolwarm
        else:
            print("Plotting customized color range")
            my_cmap, norm = color.make_cmap(clevs,neg_color=minc, pos_color=maxc)

        # Convert lat/lon to map coordinates
        x, y = m(df.XLONG[0,rs:re,cs:ce].values, df.XLAT[0,rs:re,cs:ce].values)

        # plot either contour or shading mode
        if option == "shading":
            if clevs == [-999]:
                plot_out = m.pcolormesh(x, y, a[ilev,rs:re,cs:ce], shading='auto', cmap=my_cmap)
                if cvar.any() != None:
                    plot_contour = m.contour(x, y, cvar[ilev,rs:re,cs:ce], colors=ccolor, linewidths=cthick)
            else:
                plot_out = m.pcolormesh(x, y, a[ilev,rs:re,cs:ce], cmap=my_cmap, norm=norm, shading='auto')    
                if cvar.any() != None:
                    plot_contour = m.contour(x, y, cvar[ilev,rs:re,cs:ce], levels=clevs, colors=ccolor, linewidths=cthick)
            cbar = m.colorbar(plot_out, location='bottom', pad='5%')
            cbar.set_label(labels)                            
        else:
            if clevs == [-999]:
                plot_out = m.contour(x, y, a[ilev,rs:re,cs:ce], cmap=my_cmap)
            else:
                plot_out = m.contour(x, y, a[ilev,rs:re,cs:ce], cmap=my_cmap, levels=clevs)
            plt.clabel(plot_out, inline=True, fontsize=cfont, fmt='%d')

        # adding a vector on top
        if vector:
            u1,v1 = u3d[ilev,rs:re,cs:ce],v3d[ilev,rs:re,cs:ce]
            skip = (slice(None, None, skipy), slice(None, None, skipx))  # skip 2 arrows in row, and 3 arrow in cols dim

            Q = m.quiver(x[skip], y[skip], u1[skip], v1[skip], scale=scale, color='k')
            plt.quiverkey(Q, 0.9, -0.2, 10, '10 m/s', labelpos='E')

        # Add map features
        m.drawcoastlines(linewidth=maplinewidth)
        m.drawcountries(linewidth=maplinewidth)
        m.drawparallels(np.arange(slat, elat, 10), labels=[1,0,0,0])
        m.drawmeridians(np.arange(slon, elon, 10), labels=[0,0,0,1])

        plt.title(f"{title} at level k = {ilev}")
        plt.tight_layout()
        plt.show()

    def plot_xSection_grid3D(self,df,a,xlat=-99,xlon=-99,
                             slat=-90,elat=90,
                             slon=-180,elon=180,
                             lev1=1,lev2=10,
                             clevs=[-1,0,1],ticks=[-1,0,1],
                             labels="Ticks labels",
                             title="WRF",
                             minc=(0,0,0),
                             maxc=(1,1,1),
                             zlev="m",option="shading",cfont=12):

        # locate the lat and lon idex
        rs,re,cs,ce = lat_lon_id(df,slat,elat,slon,elon)
        
        # re-define lat/lon grid start and end points
        slon_new, elon_new = df.XLONG[0,0,cs], df.XLONG[0,0,ce]
        slat_new, elat_new = df.XLAT[0,rs,0], df.XLAT[0,re,0]

        # find the id of the cross section
        if xlat != -99 and xlon != -99:
            raise ValueError('Not valid. Either xlat !=0 or xlon != 0 but cannot be both')
            return
        elif xlat !=-99:
            lat1d = df.XLAT.values[0,:,0]
            ilat = (np.abs(xlat - lat1d)).argmin() 
        elif xlon !=-99:
            lon1d = df.XLONG.values[0,0,:]
            ilon = (np.abs(xlon - lon1d)).argmin()          

        # Get the cross section slice
        if xlat !=-99:
            cross_section = a[lev1:lev2, ilat, cs:ce]  # shape = (nz, nx)
            h = df.XLONG[0,ilat,cs:ce]  
        elif xlon != -99:
            cross_section = a[lev1:lev2, rs:re, ilon]  # shape = (nz, ny)  
            h = df.XLAT[0,rs:re,ilon]  

        # define the vertical mesh
        if zlev == "m":
            z = np.linspace(0, self.nz, self.nz)   # vertical height (e.g., km)    
        elif zlev == "s": 
            z = df.ZNU.values[0]*1000
        else:
            raise ValueError('zlev option is not support. Must be level/sigma/pressure/height')
            return
        H, Z = np.meshgrid(h, z[lev1:lev2])  

        # create our own color table
        #bcols = color.myColor(clevs,minc,maxc)
        #my_cmap = ListedColormap(bcols, name='my_colormap')
        #norm = colors.BoundaryNorm(boundaries=clevs, ncolors=my_cmap.N) 

        # create my own color table
        bcols = color.myColor(clevs,minc,maxc)
        if clevs == [-999]:
            print("Plotting default color range")
            my_cmap = plt.cm.coolwarm
        else:
            print("Plotting customized color range")
            my_cmap, norm = color.make_cmap(clevs,neg_color=minc, pos_color=maxc)

        # Plot vertical cross-section 
        plt.figure(figsize=(7, 5))    
        if option == "shading":
            if clevs == [-999]:
                plot_out = plt.contourf(H, Z, cross_section, cmap='coolwarm',extend='both')
                plt.colorbar(label=labels,orientation='vertical')
            else:
                plot_out = plt.contourf(H, Z, cross_section, cmap=my_cmap, norm=norm, levels=clevs,extend='both')
                plt.colorbar(label=labels,orientation='vertical')
        elif option == "contour":
            if clevs == [-999]:
                plot_out = plt.contour(H, Z, cross_section, cmap='coolwarm')
            else:
                plot_out = plt.contour(H, Z, cross_section, cmap=my_cmap, norm=norm, levels=clevs)
            plt.clabel(plot_out, inline=True, fontsize=cfont, fmt='%d')

        # add label/title
        if xlat !=-99:
            plt.title(f'{title} at lat = {xlat}°')
            plt.xlabel('Latitude')
        elif xlon !=-99:
            plt.title(f'{title} at lon = {xlon}°')
            plt.xlabel('Longitude')    

        if zlev == "s" or zlev == "p":
            yticks = np.arange(100, 1000, 200)
            plt.ylabel('Vertical level (sigma/p)')
            plt.gca().invert_yaxis()
            plt.yscale("log")
            plt.yticks(yticks, [f"{y}" for y in yticks])
        else:
             plt.ylabel('Vertical level (k/km)')
        plt.tight_layout()
        plt.show()

# class of 2D objects to be ploted
class var2D:
    # passing a 3D dimension to initialize
    def __init__(self,nx=1,ny=1,dx=1):
        self.nx = nx
        self.ny = ny
        self.nx1 = nx+1
        self.ny1 = ny+1
        self.dx = dx

    # re-stagging U-wind component
    def shiftU2D(self,u):
        u1 = np.zeros((u.shape[1],self.ny,self.nx))
        u1[:,:] = 0.5*(u[0,:,0:self.nx]+u[0,:,1:self.nx1])
        return u1

    # re-stagging U-wind component
    def shiftV2D(self,v):
        v1 = np.zeros((v.shape[1],self.ny,self.nx))
        v1[:,:] = 0.5*(v[0,0:self.ny,:]+v[0,1:self.ny1,:])
        return v1

    # plot a single level shading on the grid (km) coordinate
    def plot_shading_grid2D(self,a,rs=1,re=2,cs=1,ce=2,
                         clevs=[-1,0,1],title="WRF",
                         minc=(0,0,0),
                         maxc=(0,0,1),
                         labels="Ticks labels",
                         ticks=[-1,0,1]):
        
        # Create contour plot
        plt.figure(figsize=(7, 5))
        Lx = (self.nx-1)*self.dx
        Ly = (self.ny-1)*self.dx
        X = np.linspace(-Lx/2, Lx/2, self.nx)   # X-axis values
        Y = np.linspace(-Ly/2, Ly/2, self.ny)   # Y-axis values
        x, y = np.meshgrid(X, Y)

        # plot color shaded using my contour levels
        bcols = color.myColor(clevs,minc,maxc)
        if re != 2 and ce !=2:
            plt.contourf(x[rs:re,cs:ce], y[rs:re,cs:ce], a[rs:re,cs:ce],
                         levels=clevs, colors=bcols)
        else:
            plt.contourf(x, y, a[ilev,:,:])

        # Add a custom colorbar
        plt.colorbar(label=labels,ticks=ticks)

        # title and axis label
        plt.title(f"{title} at surface")
        plt.xlabel("X (km)")
        plt.ylabel("Y (km)")
        plt.show()

    # plot a single level shaing on world map
    def plot_surface_world2D(self,df,a,u2d=[],v2d=[],
                    slat=-90,elat=90,
                    slon=-180,elon=180,
                    clevs=[-1,0,1],
                    title="WRF",
                    minc=(0,0,0),
                    maxc=(1,1,1),
                    labels="Ticks labels",
                    ticks=[-1,0,1],
                    scale=500,skipx=1,skipy=1,
                    option="shading",cfont=12,
                    vector=False,maplinewidth=0.5):
        
        # locate the lat and lon index
        rs,re,cs,ce = lat_lon_id(df,slat,elat,slon,elon)
        NX = ce-cs
        NY = re-rs

        # re-define lat/lon grid start and end points
        slon_new, elon_new = df.XLONG[0,0,cs], df.XLONG[0,0,ce]
        slat_new, elat_new = df.XLAT[0,rs,0], df.XLAT[0,re,0]

        # create my own color table
        if clevs == [0]:
            my_cmap = plt.cm.coolwarm
        else:
            bcols = color.myColor(clevs,minc,maxc)
            my_cmap = ListedColormap(bcols, name='my_colormap')
            norm = colors.BoundaryNorm(boundaries=clevs, ncolors=my_cmap.N) #cmap.N)

        # Create the Basemap
        fig, ax = plt.subplots(figsize=(12, 6))
        m = Basemap(projection='merc',
                    llcrnrlat=slat_new, urcrnrlat=elat_new,
                    llcrnrlon=slon_new, urcrnrlon=elon_new,
                    resolution='l', ax=ax)

        # Convert lat/lon to map coordinates
        x, y = m(df.XLONG[0,rs:re,cs:ce].values, df.XLAT[0,rs:re,cs:ce].values)

        # plot either contour or shading mode
        if option == "shading":
            if clevs == [0]:        
                plot_out = m.pcolormesh(x, y, a[rs:re,cs:ce], shading='auto', cmap=my_cmap)
            else:
                plot_out = m.pcolormesh(x, y, a[rs:re,cs:ce], cmap=my_cmap, norm=norm, shading='auto')
            cbar = m.colorbar(plot_out, location='bottom', pad='5%')
            cbar.set_label(labels)
        else:
            if clevs == [0]:
                plot_out = m.contour(x, y, a[rs:re,cs:ce], cmap=my_cmap)
            else:
                plot_out = m.contour(x, y, a[rs:re,cs:ce], cmap=my_cmap, levels=clevs)
            plt.clabel(plot_out, inline=True, fontsize=cfont, fmt='%d')

        # adding a vector on top
        if vector:
            u1,v1 = u2d[rs:re,cs:ce],v2d[rs:re,cs:ce]
            skip = (slice(None, None, skipy), slice(None, None, skipx))  # skip 2 arrows in row, and 3 arrow in cols dim

            Q = m.quiver(x[skip], y[skip], u1[skip], v1[skip], scale=scale, color='k')
            plt.quiverkey(Q, 0.9, -0.2, 10, '10 m/s', labelpos='E')

        # Add map features
        m.drawcoastlines(linewidth=maplinewidth)
        m.drawcountries(linewidth=maplinewidth)
        m.drawparallels(np.arange(slat, elat, 10), labels=[1,0,0,0])
        m.drawmeridians(np.arange(slon, elon, 10), labels=[0,0,0,1])

        plt.title(f"{title} at surface")
        plt.tight_layout()
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
