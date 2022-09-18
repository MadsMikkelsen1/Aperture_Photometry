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

# Import of FITS data
fitsURL = '/Users/madsmikkelsen/Desktop/o4201193.10.fts'
hdulist = fits.open(fitsURL)
imdata = hdulist[0].data

# Calculation of the 1st and 99th percentile value of image
L_Percent = np.percentile(imdata, 1)
U_Percent = np.percentile(imdata, 99)

# Display of image
plt.axes().set_aspect('equal')                                                             # Equal x and y axis
plt.imshow(imdata, origin = 'lower', cmap = 'viridis', clim = (L_Percent, U_Percent))      # Origin in lower left corner, colormap = viridis, limits found from 1st and 99th percentile
plt.colorbar()
plt.grid(False)
plt.show()