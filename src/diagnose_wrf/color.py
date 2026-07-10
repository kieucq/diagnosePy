import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.colors as mcolors

import numpy as np
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm

def make_cmap(clevs,
              neg_color=(0, 0, 1),    # blue RGB
              pos_color=(1, 0, 0),    # red RGB
              white=(1, 1, 1)):
    """
    Create a discrete blue-white-red colormap from contour levels.

    Parameters
    ----------
    clevs : list
        Contour levels with one interval crossing zero.
    neg_color : tuple
        RGB tuple for the darkest negative color.
    pos_color : tuple
        RGB tuple for the darkest positive color.
    white : tuple
        RGB tuple for the neutral color.

    Returns
    -------
    cmap, norm
    """

    # number of intervals
    nneg = np.sum(np.array(clevs[:-1]) < 0)
    npos = np.sum(np.array(clevs[1:]) > 0)

    # interpolate from white to endpoint colors
    neg = np.linspace(white, neg_color, nneg)
    pos = np.linspace(white, pos_color, npos)

    # remove duplicate white
    colors = np.vstack((neg[::-1], pos[1:]))

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(clevs, cmap.N)

    return cmap, norm

def myColor(clevs,startColor,endColor):
    n = len(clevs)-1
    newColor = []
    ir,ig,ib = startColor
    er,eg,eb = endColor
    dr = (er-ir)/(n-1)
    dg = (eg-ig)/(n-1)
    db = (eb-ib)/(n-1)
    for i in range(n):
        if i == 0:
            newColor.append(startColor)
        else:
            nr = ir + i*dr
            ng = ig + i*dg
            nb = ib + i*db
            newColor.append((nr,ng,nb))
    return newColor

def printColor(startColor,endColor,colors):
    print(f"Color codes from {startColor} to {endColor} are")
    for ic in colors:
        #print(f"{x:.2f}" for x in ic)
        #print(ic)
        print("({})".format(", ".join(f"{x:.2f}" for x in ic)))

def checkColorTable():
    # create my own list of color RGB codes
    clevs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    rcols = myColor(clevs,(1.,1.,1.),(1.,0.,0.))
    bcols = myColor(clevs,(1.,1.,1.),(0.,0.,1.))
    gcols = myColor(clevs,(1.,1.,1.),(0.,1.,0.))
    acols = myColor(clevs,(1.,1.,1.),(0.,1.,1.))
    ycols = myColor(clevs,(1.,1.,1.),(1.,1.,0.))
    pcols = myColor(clevs,(1.,1.,1.),(1.,0.,1.))
    hcols = [(1.0,1.0,1.0),(0.8,0.8,0.8),(0.6,0.6,0.6),
             (0.4,0.4,0.4),(0.2,0.2,0.2),(0.0,0.0,0.0)]
    print("Use the following RGB codes for manually set color (see an example of hcols in the code)")
    printColor("white","red",rcols)
    printColor("white","blue",bcols)
    printColor("white","green",gcols)
    
    # create bogus data for plotting color codes
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-X**2 - Y**2)  # Example function

    # plot contour filled with specific levels and color codes colors
    fig = plt.figure()
    plt.contourf(X, Y, Z, levels=clevs, colors=bcols)
    plt.title("Python custimized color coding using colors for each contour level")

    # colorbar with specific ticks (method 1)
    #plt.colorbar(label="Intensity").set_ticks([0.1, 0.3, 0.5, 0.9])
    plt.colorbar(label="Intensity",ticks=[0.1, 0.3, 0.5, 0.9])
    
    # plot contour filled with specific levels and color gradient cmap
    fig = plt.figure()
    redGradient = mcolors.LinearSegmentedColormap.from_list("myRed", rcols)
    plt.contourf(X, Y, Z, levels=clevs, cmap=redGradient)
    plt.title("Python custimized color coding gradient using cmap")

    # colorbar with specific ticks (method 2)
    cbar = plt.colorbar()
    cbar.set_ticks([0.1, 0.3, 0.5, 0.7])
    cbar.set_label("Intensity")

    # Create an internal Python colormap gradient from white to red to blue
    white_red_blue = mcolors.LinearSegmentedColormap.from_list("WhiteRedBlue", ["white", "red", "blue"])

    # Plot using contourf
    fig, ax = plt.subplots()
    contour = ax.contourf(X, Y, Z, levels=20, cmap=white_red_blue)
    #contour = ax.contourf(X, Y, Z, levels=[0.1, 0.3, 0.5, 0.9], cmap=white_red_blue)

    # Add colorbar
    cbar = plt.colorbar(contour)
    cbar.set_label("Intensity")
    plt.title("Python internal colormap: White to Red to Blue")

    plt.show()

    # Choose a colormap (e.g., 'viridis', 'jet', 'Reds', or a custom one) to print out RGB codes
    cmap_name = 'bwr'  # Change to any colormap you like
    cmap = plt.get_cmap(cmap_name, 20)
    print(f"20 RGB codes for cmap color {cmap_name} are:")
    for i in range(cmap.N):
        rgb = cmap(i)  # Get the RGBA tuple (Red, Green, Blue, Alpha)
        print(f"Value {i}: RGB = {rgb[:3]}")  # Ignore Alpha (transparency)

#checkColorTable()
