# APERTURE PHOTOMETRY #

# Import of various libaries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.io import fits
from matplotlib.widgets import Slider, Button

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

# Circles
""" 
Description of circles. 
    Circle 1: Creates a circle around the star with a custom radius appropriate to the star's apparent size.
    Circle 2: Creates a circle with minimum radius of circle 1. This is the lower bound for the background noise collection.
"""
# Defining circle 1
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

# Defining circle 2
def createCircle2(x, y, r):
    x_cent = x
    y_cent = y
    radius = r
    circle2 = plt.Circle((x, y),
                        radius = radius,
                        color = 'b',
                        fill = False,
                        lw = 2)
    return circle2

# Coordinates from mouse click
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
            distance = np.sqrt( (i - x)**2 + (j - y)**2 )               # Distance
            if distance < r:
                PixCollec = np.append(PixCollec, imdata[j][i])
    C1 = np.sum(PixCollec)
    A1 = len(PixCollec)

    return PixCollec

poi_inner_pixels = pixel_collector(214, 239, 8)                # Counting pixels inside inner-most circle for Point of Interest star
poi_innertorus_pixels = pixel_collector(214, 239, 2*8)         # Counting pixels inside inner-most torus circle for Point of Interest star
poi_outertorus_pixels = pixel_collector(214, 239, 3*8)         # Counting pixels from outer-most torus circle for POI star
poi_background_noise = np.sum(poi_outertorus_pixels) - np.sum(poi_innertorus_pixels)                 # Making sure only pixels inside torus is calculated
poi_star_brightness = np.sum(poi_inner_pixels)                                                       # Star of interest brightness

# Pixel collector within circles
def refstar_pixel_collector(x, y, r):
    refstar_PixCollec = np.array([])
    refstar_x_cent = x
    refstar_y_cent = y
    refstar_radius = r
    for i in range(x - r, x + r):                                       # Pixel collector in x-direction 
        for j in range(y - r, y + r):                                   # Picel collector in y-direction
            distance = np.sqrt( (i - x)**2 + (j - y)**2 )               # Distance
            if distance < r:
                refstar_PixCollec = np.append(refstar_PixCollec, imdata[j][i])
    refstar_C1 = np.sum(refstar_PixCollec)
    refstar_A1 = len(refstar_PixCollec)

    return refstar_PixCollec

refstar_inner_pixels = refstar_pixel_collector(181, 268, 8)                # Counting pixels inside inner-most circle for Point of Interest star
refstar_innertorus_pixels = refstar_pixel_collector(181, 268, 2*8)         # Counting pixels inside inner-most torus circle for Point of Interest star
refstar_outertorus_pixels = refstar_pixel_collector(181, 268, 3*8)         # Counting pixels from outer-most torus circle for POI star
refstar_background_noise = np.sum(refstar_outertorus_pixels) - np.sum(refstar_innertorus_pixels)     # Making sure only pixels within torus is calculated
refstar_star_brightness = np.sum(refstar_inner_pixels)                                               # Star brightness 
refstar_inner_area = len(refstar_inner_pixels)
refstar_torus_area = len(refstar_outertorus_pixels) - len(refstar_innertorus_pixels)

print("Inner area: %.f pixels" % refstar_inner_area)
print("Torus area: %.f pixels" % refstar_torus_area)

# Printing values for brightness and background noise
print("Brightness for star of interest: %.f" % poi_star_brightness)
print("Brightness for reference star: %.f" % refstar_star_brightness)
print("Background noise around star of interest: %.f" % poi_background_noise)
print("Background noise around reference star: %.f" % refstar_background_noise)

# Calculation of brightness within inner and outer circle
def brightness(C1, C2, A1, A2):
    l = C2 - ((C1 - C2) / (A1 - A2)) * A2

    return l

print("Brightness of V\u2081: %.f" % brightness(poi_star_brightness, poi_background_noise, refstar_inner_area , refstar_torus_area))          # Result from brightness formula using C1, C2, A1, A2 from 'PixCollec' function
print("Brightness of S\u2081: %.f" % brightness(refstar_star_brightness, refstar_background_noise, refstar_inner_area, refstar_torus_area))
l1 = brightness(poi_star_brightness, poi_background_noise, refstar_inner_area , refstar_torus_area) 
l2 = brightness(refstar_star_brightness, refstar_background_noise, refstar_inner_area, refstar_torus_area) 

# Display of image
plt.axes().set_aspect('equal')                                                             # Equal x and y axis
plt.imshow(imdata, origin = 'lower', cmap = 'gray', clim = (L_Percent, U_Percent))         # Origin in lower left corner, colormap, limits found from 1st and 99th percentile
plt.colorbar(label = 'Intensity')
plt.grid(False)

# Display of star of interest circles
poi_inner = createCircle(214, 239, 8)               # Drawing inner circle for point of interest star at (x_coordinate, y_coordinate, radius_inner)
poi_innertorus = createCircle(214, 239, 2*8)        # Drawing inner torus circles for POI star at (x_coordinate, y_coordinate, 2*radius_inner)
poi_outertorus = createCircle(214, 239, 3*8)        # Drawing outer torus circle for POI star at (x_coordinate, y_coordinate, 3*radius_inner)
showCircle(poi_inner)
showCircle(poi_innertorus)
showCircle(poi_outertorus)
plt.scatter(214, 239, marker = '+', s = 20, c = 'r')

# Display of reference star circles
refstar_inner = createCircle2(181, 268, 8)            # Drawing inner circle for reference star at (x_coordinate, y_coordinate, radius_inner)
refstar_innertorus = createCircle2(181, 268, 2*8)      # Drawing outer circle for reference star at (x_coordinate, y_coordinate, 2*radius_inner)
refstar_outertorus = createCircle2(181, 268, 3*8)      # Drawing outer circle for reference star at (x_coordinate, y_coordinate, 3*radius_inner)
showCircle(refstar_inner)
showCircle(refstar_innertorus)
showCircle(refstar_outertorus)
plt.scatter(181, 268, marker = '+', s = 20, c = 'b')

# Calculation of magnitudes
def mag_calc(l1, l2):
    l1 = l1
    l2 = l2
    magnitude = -2.5 * np.log(l1 / l2)

    return magnitude

print("Magnitude ratio: %.3F" % mag_calc(l1, l2))

#Plot af brightness af V_1 som funktion af blænderadius
#l_list = np.array([0, 34716, 134976, 187855, 186337, 3086240, 207800])
#r_list = np.array([0, 2, 4, 6, 8, 10, 12])

#plt.scatter(r_list, l_list)

#   Defining Slider
#t = np.linspace(0, 1, 1000)

# Define initial parameters
#init_radius = 5

# Make a vertically oriented slider to control the radius
#axrad = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
#amp_slider = Slider(
#    ax=axrad,
#    label="Radius",
#    valmin=0.5,
#    valmax=20,
#    valinit=init_radius,
#    orientation="vertical"
#)

# The function to be called anytime a slider's value changes
#def update(val):
#    plt.Circle.set_radius(createCircle(t, amp_slider.val, amp_slider.val))
#    fig.canvas.draw_idle()

# register the update function with each slider
#amp_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
#resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
#button = Button(resetax, 'Reset', hovercolor='0.975')


#def reset(event):
#    amp_slider.reset()
#button.on_clicked(reset)


plt.show()