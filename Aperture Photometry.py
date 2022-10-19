# APERTURE PHOTOMETRY #
#%%
# Import of various libaries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.io import fits
import glob
fitsfiles = glob.glob('./mh*.fit')

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
fitsURL = 'o4201193.10.fts'
hdulist = fits.open(fitsURL)
hdulist[0].header
imdata = hdulist[0].data
exptime = hdulist[0].header["EXPTIME"]   # Exposure time [sec] (150)
gain = hdulist[0].header["GAIN"]         # Readout gain [e/ADU] (1.8)
rdnoise = hdulist[0].header["RDNOISE"]   # Readout noise [e] (5.1)
airmass = hdulist[0].header["AIRMASS"]   # Airmass (1.378)
heljd = hdulist[0].header["HELJD"]       # Heliocentric Julian Date of mid exposure (2455950.638041)
#print(exptime, gain, rdnoise, airmass, heljd)


# Calculation of the 5th and 95th percentile value of image
L_Percent = np.percentile(imdata, 10)
U_Percent = np.percentile(imdata, 95)

# Display of image
plt.axes().set_aspect('equal')                                                             # Equal x and y axis
plt.grid(False)
plt.imshow(imdata, origin = 'lower', cmap = 'gray', clim = (L_Percent, U_Percent))         # Origin in lower left corner, colormap, limits found from 1st and 99th percentile

# Star of interest
X_STAR1 = 214
Y_STAR1 = 239

# Reference star
X_STAR2 = 181
Y_STAR2 = 267

# Aperature Radius
ap_rad = 7

# Circles
# Defining circle
def createCircle(x, y, r, c):
    x_cent = x
    y_cent = y
    radius = r
    circle = plt.Circle((x, y),
                        radius = r,
                        color = c,
                        fill = False,
                        lw = 2)
    return circle

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
            if distance < r:                                            # Count pixels if within inner radius
                PixCollec = np.append(PixCollec, imdata[j][i])
    C1 = np.sum(PixCollec)
    A1 = len(PixCollec)

    return PixCollec

poi_inner_pixels = pixel_collector(X_STAR1, Y_STAR1, ap_rad)                # Counting pixels inside inner-most circle for Point of Interest star
poi_innertorus_pixels = pixel_collector(X_STAR1, Y_STAR1, 2*ap_rad)         # Counting pixels inside inner-most torus circle for Point of Interest star
poi_outertorus_pixels = pixel_collector(X_STAR1, Y_STAR1, 3*ap_rad)         # Counting pixels from outer-most torus circle for POI star
poi_background_noise = np.sum(poi_outertorus_pixels) - np.sum(poi_innertorus_pixels)                 # Making sure only pixels inside torus is calculated
poi_star_brightness = np.sum(poi_inner_pixels)
poi_inner_area = len(poi_inner_pixels)
poi_torus_area = len(poi_outertorus_pixels) - len(poi_innertorus_pixels)                                                      # Star of interest brightness

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

refstar_inner_pixels = refstar_pixel_collector(X_STAR2, Y_STAR2, ap_rad)                # Counting pixels inside inner-most circle for Point of Interest star
refstar_innertorus_pixels = refstar_pixel_collector(X_STAR2, Y_STAR2, 2*ap_rad)         # Counting pixels inside inner-most torus circle for Point of Interest star
refstar_outertorus_pixels = refstar_pixel_collector(X_STAR2, Y_STAR2, 3*ap_rad)         # Counting pixels from outer-most torus circle for POI star
refstar_background_noise = np.sum(refstar_outertorus_pixels) - np.sum(refstar_innertorus_pixels)     # Making sure only pixels within torus is calculated
refstar_star_brightness = np.sum(refstar_inner_pixels)                                               # Star brightness 
refstar_inner_area = len(refstar_inner_pixels)
refstar_torus_area = len(refstar_outertorus_pixels) - len(refstar_innertorus_pixels)

print("Inner area: %.f pixels" % poi_inner_area)
print("Torus area: %.f pixels" % poi_torus_area)

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

# Display of star of interest circles
poi_inner = createCircle(X_STAR1, Y_STAR1, ap_rad, 'r')               # Drawing inner circle for point of interest star at (x_coordinate, y_coordinate, radius_inner)
poi_innertorus = createCircle(X_STAR1, Y_STAR1, 2*ap_rad, 'r')        # Drawing inner torus circles for POI star at (x_coordinate, y_coordinate, 2*radius_inner)
poi_outertorus = createCircle(X_STAR1, Y_STAR1, 3*ap_rad, 'r')        # Drawing outer torus circle for POI star at (x_coordinate, y_coordinate, 3*radius_inner)
showCircle(poi_inner)
showCircle(poi_innertorus)
showCircle(poi_outertorus)
plt.scatter(X_STAR1, Y_STAR1, marker = '+', s = 20, c = 'r')

# Display of reference star circles
refstar_inner = createCircle(X_STAR2, Y_STAR2, ap_rad, 'g')             # Drawing inner circle for reference star at (x_coordinate, y_coordinate, radius_inner, colour)
refstar_innertorus = createCircle(X_STAR2, Y_STAR2, 2*ap_rad, 'g')      # Drawing outer circle for reference star at (x_coordinate, y_coordinate, 2*radius_inner, colour)
refstar_outertorus = createCircle(X_STAR2, Y_STAR2, 3*ap_rad, 'g')      # Drawing outer circle for reference star at (x_coordinate, y_coordinate, 3*radius_inner, colour)
showCircle(refstar_inner)
showCircle(refstar_innertorus)
showCircle(refstar_outertorus)
plt.scatter(X_STAR2, Y_STAR2, marker = '+', s = 20, c = 'g')

# Counting pixels in x and y direction near star of interest
delta = 100                              # Size of cutout around star
xpixels = np.arange(X_STAR1 - delta, X_STAR1 + delta)
ypixels = np.arange(Y_STAR1 - delta, Y_STAR1 + delta)

plt.figure()
plt.xlabel('Pixels')
plt.ylabel('Counts')
plt.plot(xpixels, imdata[Y_STAR1, xpixels], label='x')
plt.plot(ypixels, imdata[xpixels, X_STAR1], label='y')
plt.legend()
# Calculation of magnitudes
def mag_calc(l1, l2):
    magnitude = -2.5 * np.log(l1 / l2)

    return magnitude

def star_brightness_L(x_star, y_star):
    L = []
    for i in range(0, 25):    
        poi_inner_pixels = pixel_collector(x_star, y_star, i)                # Counting pixels inside inner-most circle for Point of Interest star
        poi_innertorus_pixels = pixel_collector(x_star, y_star, 2*i)         # Counting pixels inside inner-most torus circle for Point of Interest star
        poi_outertorus_pixels = pixel_collector(x_star, y_star, 3*i)         # Counting pixels from outer-most torus circle for POI star
        poi_background_noise = np.sum(poi_outertorus_pixels) - np.sum(poi_innertorus_pixels)                 # Making sure only pixels inside torus is calculated
        poi_star_brightness = np.sum(poi_inner_pixels)                                                       # Star of interest brightness
        L.append(poi_star_brightness)
    return L
    print(L)

l_list_test = star_brightness_L(X_STAR1, Y_STAR1)
#print(l_list_test)

print("Magnitude ratio: %.3F" % mag_calc(l1, l2))

# Cropped image around star of interest
xnew, ynew = np.meshgrid(xpixels, ypixels)
crop = imdata[ynew, xnew]
plt.figure()
plt.imshow(crop, origin = 'lower', cmap = 'gray', clim = (L_Percent, U_Percent))         # Origin in lower left corner, colormap, limits found from 1st and 99th percentile
plt.grid(False)

showCircle(createCircle(100, 99, ap_rad, 'r'))
showCircle(createCircle(100, 99, 2*ap_rad, 'r'))
showCircle(createCircle(100, 99, 3*ap_rad, 'r'))
plt.scatter(100, 99, marker = '+', s = 20, c = 'r')

showCircle(createCircle(67, 128, ap_rad, 'g'))
showCircle(createCircle(67, 128, 2*ap_rad, 'g'))
showCircle(createCircle(67, 128, 3*ap_rad, 'g'))
plt.scatter(67, 128, marker = '+', s = 20, c = 'g')

distance = np.sqrt( (X_STAR1 - xnew)**2 + (Y_STAR1 - ynew)**2)

background = crop[(distance > 2*ap_rad) & (distance < 3*ap_rad)]
background_flux = np.sum(background)
background_dens = np.median(background)

# Cropped image and background noise subtracted
plt.figure()
plt.imshow(imdata[ynew, xnew] - background_dens, \
    vmax = np.percentile(imdata[ynew, xnew] - background_dens, 95), \
    vmin = np.percentile(imdata[ynew, xnew] - background_dens, 10),  
    origin = 'lower',
    cmap = 'gray')
plt.grid(False)

showCircle(createCircle(100, 99, ap_rad, 'r'))
showCircle(createCircle(100, 99, 2*ap_rad, 'r'))
showCircle(createCircle(100, 99, 3*ap_rad, 'r'))
plt.scatter(100, 99, marker = '+', s = 20, c = 'r')

showCircle(createCircle(67, 128, ap_rad, 'g'))
showCircle(createCircle(67, 128, 2*ap_rad, 'g'))
showCircle(createCircle(67, 128, 3*ap_rad, 'g'))
plt.scatter(67, 128, marker = '+', s = 20, c = 'g')

# Calculating the signal-noise ration
N_pix = len(crop[poi_inner_area])
N_obj = np.sum(crop[distance < ap_rad]) - background_dens * N_pix
signal_noise = (gain * N_obj) / (np.sqrt(gain * N_obj + N_pix * gain * background_dens + N_pix * rdnoise**2)) 
poi_magnitude = 25 - 2.5 * np.log10(N_obj)

print('Signal-to-noise ratio for an aperture size of {} is {:0.2f}'.format(ap_rad, signal_noise))
print('Magnitude for an aperture size of {} is {:0.2f}'.format(ap_rad, poi_magnitude))

# Plotting for various radii
magni_func = lambda n: 25 - 2.5 * np.log10(n)

signal_noise_func = lambda n, pixels, sky: (gain * n) / (np.sqrt(gain * n + pixels * gain * sky + pixels * rdnoise**2)) 

possible_ap_radii = np.arange(1, 39)
magnis = np.zeros(len(possible_ap_radii))
sig_noi = np.zeros(len(possible_ap_radii))

for i, ap_rad in enumerate(possible_ap_radii):
    inner = distance < ap_rad
    poi = crop[inner]
    N_pix = len(poi)
    N_obj = np.sum(poi) - background_dens * N_pix
    magnis[i] = magni_func(N_obj)
    sig_noi[i] = signal_noise_func(N_obj, N_pix, background_dens) 

fig, ax = plt.subplots()
ax.set_xlabel("Aperture radius")
ax.set_ylabel("Magnitude")
ax.invert_yaxis()
ax.plot(possible_ap_radii, magnis)

fig, ax = plt.subplots()
ax.set_xlabel("Aperture radius")
ax.set_ylabel("Signal-to-noise ratio")
ax.scatter(7, max(sig_noi), marker = 'x', color = 'black', zorder = 3)
ax.hlines(max(sig_noi), 3, 11, color = 'black', ls = "dotted")
ax.vlines(7, 0, max(sig_noi), color = 'black', ls = 'dotted')
ax.text(11.5, 473, '(7, 481)')
ax.plot(possible_ap_radii, sig_noi)

print(np.interp(437.47, sig_noi, possible_ap_radii))
plt.show()
