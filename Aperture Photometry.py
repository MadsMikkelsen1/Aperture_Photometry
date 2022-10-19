# APERTURE PHOTOMETRY #
#%%
# Import of various libaries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.io import fits
import glob

# Ignore all that numpy error shit
np.seterr(divide='ignore', invalid='ignore')

class Photometry:
    def __init__(self, fitsURL, offset_x, offset_y, show_loaded=True):
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
        self.mycolors = ['#C188F7','#F79288','#7FB806','#F59B18']
        sns.set_palette("Set2") 
        self.fig = plt.gcf()

        # Import of FITS data
        hdulist = fits.open(fitsURL)
        hdulist[0].header
        self.imdata = hdulist[0].data
        self.exptime = hdulist[0].header["EXPTIME"]   # Exposure time [sec] (150)
        self.gain = hdulist[0].header["GAIN"]         # Readout gain [e/ADU] (1.8)
        self.rdnoise = hdulist[0].header["RDNOISE"]   # Readout noise [e] (5.1)
        self.airmass = hdulist[0].header["AIRMASS"]   # Airmass (1.378)
        self.heljd = hdulist[0].header["HELJD"]       # Heliocentric Julian Date of mid exposure (2455950.638041)
        #print(exptime, gain, rdnoise, airmass, heljd)


        # Calculation of the 5th and 95th percentile value of image
        self.L_Percent = np.percentile(self.imdata, 10)
        self.U_Percent = np.percentile(self.imdata, 95)

        if show_loaded:
            # Display of image
            plt.axes().set_aspect('equal')                                                             # Equal x and y axis
            plt.grid(False)
            plt.imshow(self.imdata, origin = 'lower', cmap = 'gray', clim = (self.L_Percent, self.U_Percent))         # Origin in lower left corner, colormap, limits found from 1st and 99th percentile

        # Stars of interest
        self.X_STAR1 = 214 + offset_x
        self.Y_STAR1 = 239 + offset_y

        self.X_STAR2 = 181 + offset_x
        self.Y_STAR2 = 267 + offset_y

    # Circles
    # Defining circle
    def createCircle(self, x, y, r, c):
        self.x_cent = x
        self.y_cent = y
        self.radius = r
        circle = plt.Circle((x, y),
                            radius = r,
                            color = c,
                            fill = False,
                            lw = 2)
        return circle

    # Coordinates from mouse click
    def onclick(self, event):
        if event.xdata != None and event.ydata != None:
            x_click = event.xdata
            y_click = event.ydata
            print(event.xdata, event.ydata)

        return x_click, y_click

    # Adding artist object onto figure
    def showCircle(self, patch):
        ax = plt.gca()
        ax.add_patch(patch)

        plt.axis('scaled')

    # Pixel collector within circles
    def pixel_collector(self, x, y, r):
        PixCollec = np.array([])
        x_cent = x
        y_cent = y
        radius = r
        for i in range(x - r, x + r):                                       # Pixel collector in x-direction 
            for j in range(y - r, y + r):                                   # Picel collector in y-direction
                distance = np.sqrt( (i - x)**2 + (j - y)**2 )               # Distance
                if distance < r:
                    PixCollec = np.append(PixCollec, self.imdata[j][i])
        C1 = np.sum(PixCollec)
        A1 = len(PixCollec)

        return PixCollec

    # Pixel collector within circles
    def refstar_pixel_collector(self, x, y, r):
        refstar_PixCollec = np.array([])
        self.refstar_x_cent = x
        self.refstar_y_cent = y
        self.refstar_radius = r
        for i in range(x - r, x + r):                                       # Pixel collector in x-direction 
            for j in range(y - r, y + r):                                   # Picel collector in y-direction
                distance = np.sqrt( (i - x)**2 + (j - y)**2 )               # Distance
                if distance < r:
                    refstar_PixCollec = np.append(refstar_PixCollec, self.imdata[j][i])
        self.refstar_C1 = np.sum(refstar_PixCollec)
        self.refstar_A1 = len(refstar_PixCollec)

        return refstar_PixCollec

    # Calculation of brightness within inner and outer circle
    def brightness(self, C1, C2, A1, A2):
        l = C2 - ((C1 - C2) / (A1 - A2)) * A2

        return l

    # Calculation of magnitudes
    def mag_calc(self, l1, l2):
        magnitude = -2.5 * np.log(l1 / l2)

        return magnitude

    def star_brightness_L(self, x_star, y_star):
        L = []
        for i in range(0, 25):    
            poi_inner_pixels = self.pixel_collector(x_star, y_star, i)                # Counting pixels inside inner-most circle for Point of Interest star
            poi_innertorus_pixels = self.pixel_collector(x_star, y_star, 2*i)         # Counting pixels inside inner-most torus circle for Point of Interest star
            poi_outertorus_pixels = self.pixel_collector(x_star, y_star, 3*i)         # Counting pixels from outer-most torus circle for POI star
            poi_background_noise = np.sum(poi_outertorus_pixels) - np.sum(poi_innertorus_pixels)                 # Making sure only pixels inside torus is calculated
            poi_star_brightness = np.sum(poi_inner_pixels)                                                       # Star of interest brightness
            L.append(poi_star_brightness)
        return L

    def do_calculation(self, show=True):
        cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        ap_rad = 8
        poi_inner_pixels = self.pixel_collector(self.X_STAR1, self.Y_STAR1, ap_rad)               # Counting pixels inside inner-most circle for Point of Interest star
        poi_innertorus_pixels = self.pixel_collector(self.X_STAR1, self.Y_STAR1, 2*ap_rad)         # Counting pixels inside inner-most torus circle for Point of Interest star
        poi_outertorus_pixels = self.pixel_collector(self.X_STAR1, self.Y_STAR1, 3*ap_rad)         # Counting pixels from outer-most torus circle for POI star
        poi_background_noise = np.sum(poi_outertorus_pixels) - np.sum(poi_innertorus_pixels)                 # Making sure only pixels inside torus is calculated
        poi_star_brightness = np.sum(poi_inner_pixels)
        poi_inner_area = len(poi_inner_pixels)
        poi_torus_area = len(poi_outertorus_pixels) - len(poi_innertorus_pixels)                                                      # Star of interest brightness
    

        refstar_inner_pixels = self.refstar_pixel_collector(self.X_STAR2, self.Y_STAR2, ap_rad)                # Counting pixels inside inner-most circle for Point of Interest star
        refstar_innertorus_pixels = self.refstar_pixel_collector(self.X_STAR2, self.Y_STAR2, 2*ap_rad)         # Counting pixels inside inner-most torus circle for Point of Interest star
        refstar_outertorus_pixels = self.refstar_pixel_collector(self.X_STAR2, self.Y_STAR2, 3*ap_rad)         # Counting pixels from outer-most torus circle for POI star
        refstar_background_noise = np.sum(refstar_outertorus_pixels) - np.sum(refstar_innertorus_pixels)     # Making sure only pixels within torus is calculated
        refstar_star_brightness = np.sum(refstar_inner_pixels)                                               # Star brightness 
        refstar_inner_area = len(refstar_inner_pixels)
        refstar_torus_area = len(refstar_outertorus_pixels) - len(refstar_innertorus_pixels)

        if show:
            print("Inner area: %.f pixels" % poi_inner_area)
            print("Torus area: %.f pixels" % poi_torus_area)

            # Printing values for brightness and background noise
            print("Brightness for star of interest: %.f" % poi_star_brightness)
            print("Brightness for reference star: %.f" % refstar_star_brightness)
            print("Background noise around star of interest: %.f" % poi_background_noise)
            print("Background noise around reference star: %.f" % refstar_background_noise)
            print("Brightness of V\u2081: %.f" % self.brightness(poi_star_brightness, poi_background_noise, refstar_inner_area , refstar_torus_area))          # Result from brightness formula using C1, C2, A1, A2 from 'PixCollec' function
            print("Brightness of S\u2081: %.f" % self.brightness(refstar_star_brightness, refstar_background_noise, refstar_inner_area, refstar_torus_area))
            l1 = self.brightness(poi_star_brightness, poi_background_noise, refstar_inner_area , refstar_torus_area) 
            l2 = self.brightness(refstar_star_brightness, refstar_background_noise, refstar_inner_area, refstar_torus_area) 

            # Display of star of interest circles
            poi_inner = self.createCircle(self.X_STAR1, self.Y_STAR1, ap_rad, 'r')               # Drawing inner circle for point of interest star at (x_coordinate, y_coordinate, radius_inner)
            poi_innertorus = self.createCircle(self.X_STAR1, self.Y_STAR1, 2*ap_rad, 'r')        # Drawing inner torus circles for POI star at (x_coordinate, y_coordinate, 2*radius_inner)
            poi_outertorus = self.createCircle(self.X_STAR1, self.Y_STAR1, 3*ap_rad, 'r')        # Drawing outer torus circle for POI star at (x_coordinate, y_coordinate, 3*radius_inner)
            self.showCircle(poi_inner)
            self.showCircle(poi_innertorus)
            self.showCircle(poi_outertorus)
            plt.scatter(self.X_STAR1, self.Y_STAR1, marker = '+', s = 20, c = 'r')

            # Display of reference star circles
            refstar_inner = self.createCircle(self.X_STAR2, self.Y_STAR2, ap_rad, 'g')             # Drawing inner circle for reference star at (x_coordinate, y_coordinate, radius_inner, colour)
            refstar_innertorus = self.createCircle(self.X_STAR2, self.Y_STAR2, 2*ap_rad, 'g')      # Drawing outer circle for reference star at (x_coordinate, y_coordinate, 2*radius_inner, colour)
            refstar_outertorus = self.createCircle(self.X_STAR2, self.Y_STAR2, 3*ap_rad, 'g')      # Drawing outer circle for reference star at (x_coordinate, y_coordinate, 3*radius_inner, colour)
            self.showCircle(refstar_inner)
            self.showCircle(refstar_innertorus)
            self.showCircle(refstar_outertorus)
            plt.scatter(self.X_STAR2, self.Y_STAR2, marker = '+', s = 20, c = 'g')

        # Counting pixels in x and y direction near star of interest
        delta = 100                              # Size of cutout around star
        xpixels = np.arange(self.X_STAR1 - delta, self.X_STAR1 + delta)
        ypixels = np.arange(self.Y_STAR1 - delta, self.Y_STAR1 + delta)
        if show:
            fig, ax = plt.subplots()
            ax.set_xlabel('Pixels')
            ax.set_ylabel('Counts')
            ax.plot(xpixels, self.imdata[self.Y_STAR1, xpixels], label='x')
            ax.plot(ypixels, self.imdata[xpixels, self.X_STAR1], label='y')
            ax.legend()

            l_list_test = self.star_brightness_L(self.X_STAR1, self.Y_STAR1)
            #print(l_list_test)

            print("Magnitude ratio: %.3F" % self.mag_calc(l1, l2))

            # Cropped image around star of interest
        xnew, ynew = np.meshgrid(xpixels, ypixels)

        crop = self.imdata[ynew, xnew]
        if show:
            plt.figure(10)
            plt.imshow(crop, origin = 'lower', cmap = 'gray', clim = (self.L_Percent, self.U_Percent))         # Origin in lower left corner, colormap, limits found from 1st and 99th percentile
            plt.grid(False)

            self.showCircle(self.createCircle(100, 99, ap_rad, 'r'))
            self.showCircle(self.createCircle(100, 99, 2*ap_rad, 'r'))
            self.showCircle(self.createCircle(100, 99, 3*ap_rad, 'r'))
            plt.scatter(100, 99, marker = '+', s = 20, c = 'r')

            self.showCircle(self.createCircle(67, 128, ap_rad, 'g'))
            self.showCircle(self.createCircle(67, 128, 2*ap_rad, 'g'))
            self.showCircle(self.createCircle(67, 128, 3*ap_rad, 'g'))
            plt.scatter(67, 128, marker = '+', s = 20, c = 'g')

        distance = np.sqrt( (self.X_STAR1 - xnew)**2 + (self.Y_STAR1 - ynew)**2)

        background = crop[(distance > 2*ap_rad) & (distance < 3*ap_rad)]
        background_flux = np.sum(background)
        background_dens = np.median(background)

        if show:
            # Cropped image and background noise subtracted
            plt.figure(11)
            plt.imshow(self.imdata[ynew, xnew] - background_dens, \
                vmax = np.percentile(self.imdata[ynew, xnew] - background_dens, 95), \
                vmin = np.percentile(self.imdata[ynew, xnew] - background_dens, 10),  
                origin = 'lower',
                cmap = 'gray')
            plt.grid(False)

            self.showCircle(self.createCircle(100, 99, ap_rad, 'r'))
            self.showCircle(self.createCircle(100, 99, 2*ap_rad, 'r'))
            self.showCircle(self.createCircle(100, 99, 3*ap_rad, 'r'))
            plt.scatter(100, 99, marker = '+', s = 20, c = 'r')

            self.showCircle(self.createCircle(67, 128, ap_rad, 'g'))
            self.showCircle(self.createCircle(67, 128, 2*ap_rad, 'g'))
            self.showCircle(self.createCircle(67, 128, 3*ap_rad, 'g'))
            plt.scatter(67, 128, marker = '+', s = 20, c = 'g')

        # Calculating the signal-noise ratio
        N_pix = len(crop[poi_inner_area])
        N_obj = np.sum(crop[distance < ap_rad]) - background_dens * N_pix
        signal_noise = (self.gain * N_obj) / (np.sqrt(self.gain * N_obj + N_pix * self.gain * background_dens + N_pix * self.rdnoise**2)) 
        poi_magnitude = 25 - 2.5 * np.log10(N_obj)

        if show:
            print('Signal-to-noise ratio for an aperture size of {} is {:0.2f}'.format(ap_rad, signal_noise))
            print('Magnitude for an aperture size of {} is {:0.2f}'.format(ap_rad, poi_magnitude))

        # Plotting for various radii
        magni_func = lambda n: 25 - 2.5 * np.log10(n)

        signal_noise_func = lambda n, pixels, sky: (self.gain * n) / (np.sqrt(self.gain * n + pixels * self.gain * sky + pixels * self.rdnoise**2)) 

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

        if show:
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

        if show:
            plt.show()
        
        return (N_obj, self.heljd)

if __name__ == "__main__": 
    offsets = np.genfromtxt("Cepheider - fits/offsets.txt", dtype=int)
    fitsfiles = glob.glob('Cepheider - fits/*.fts')

    print("Processing images!")

    N_objs = []
    heljds = []

    for (off_x, off_y), fileurl in zip(offsets, fitsfiles):
        show_text = f"Processing file {fileurl}"
        print(show_text)
        print("-"*len(show_text))

        p = Photometry(fileurl, off_x, off_y, show_loaded=False)
        N_obj, heljd = p.do_calculation(show=False)

        N_objs.append(N_obj)
        N_objs_new = np.where(np.array(N_objs) < 1e6, N_objs,0)
        heljds.append(heljd)

    plt.scatter(heljds, N_objs_new)
    plt.show()