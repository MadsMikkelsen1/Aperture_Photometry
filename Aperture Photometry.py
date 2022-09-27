# APERTURE PHOTOMETRY #

# Import of various libaries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.io import fits

# Matplotlib and Seaborn plot customizations
plt.rcParams['figure.figsize'] = (15,10)
plt.rc("axes", labelsize=18)
plt.rc("xtick", labelsize=16, top=True, direction="in")
plt.rc("ytick", labelsize=16, right=True, direction="in")
plt.rc("axes", titlesize=22)
plt.rc("legend", fontsize=16, loc="upper left")
plt.rc("figure", figsize=(10, 7))
sns.set_style("darkgrid", {'axes.grid' : False})
sns.set_context("paper")
sns.set(font_scale=1.4)
mycolors = ['#C188F7','#F79288','#7FB806','#F59B18']
sns.set_palette("Set2") 
fig = plt.gcf()

# Import of FITS data
fitsURL = '/Users/madsmikkelsen/Desktop/o4201193.10.fts'
hdulist = fits.open(fitsURL)
imdata = hdulist[0].data

# Calculation of the 1st and 99th percentile value of image
L_Percent = np.percentile(imdata, 1)
U_Percent = np.percentile(imdata, 99)


# Defining circle
def createCircle(x, y, r):
    x_cent = x
    y_cent = y
    radius = r
    circle = plt.Circle((x, y),
                        radius = radius,
                        color = 'r',
                        fill = False,
                        lw = 2)
    return circle

def onclick(event):
    if event.xdata != None and event.ydata != None:
        x_click = event.xdata
        y_click = event.ydata
        print(event.xdata, event.ydata)

    return x_click, y_click
cid = fig.canvas.mpl_connect('button_press_event', onclick)

# Adding artist object onto figure
def showCircle(patch):
    ax = plt.gca()
    ax.add_patch(patch)

    plt.axis('scaled')

# Pixel collector within circles
def pixel_collector(x, y, r):
    PixCollec = np.array([])
    x_cent = x
    y_cent = y
    radius = r
    for i in range(x - r, x + r):                                       # Pixel collector in x-direction 
        for j in range(y - r, y + r):                                   # Picel collector in y-direction
            distance = np.sqrt((i - x)**2 + (j - y)**2 )                # Distance
            if distance < r:
                PixCollec = np.append(PixCollec, imdata[j][i])
    C1 = np.sum(PixCollec)
    A1 = len(PixCollec)

    print(C1, A1)
    return PixCollec

pixel_collector(181, 266, 15)           # Inner circle 
pixel_collector(181, 266, 2*15)         # Outer circle

# Calculation of brightness within inner and outer circle
def brightness(C1, C2, A1, A2):
    l = C2 - ((C1 - C2) / (A1 - A2)) * A2

    print(l)
    return l

brightness(993165, 427650, 697, 2809)     # Result from brightness formula using A1, A2, C1, C2 from 'PixCollec' function

# Display of image
plt.axes().set_aspect('equal')                                                             # Equal x and y axis
plt.imshow(imdata, origin = 'lower', cmap = 'gray', clim = (L_Percent, U_Percent))         # Origin in lower left corner, colormap, limits found from 1st and 99th percentile
plt.colorbar(label = 'Intensity')
plt.grid(False)

c_inner = createCircle(181, 266, 15)               # Drawing inner circle at (x_coordinate, y_coordinate, radius_inner)
c_outer = createCircle(181, 266, 2*15)             # Drawing outer circle at (x_coordinate, y_coordinate, 2*radius_inner)
showCircle(c_inner)
showCircle(c_outer)

def mag_calc(l1, l2):
    l1 = l1
    l2 = l2
    magnitude = -2.5*np.log(l1 / l2)

return magnitude

plt.show()