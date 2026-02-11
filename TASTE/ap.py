import numpy as np
import pickle 
from astropy.io import fits
from astropy import coordinates as coord, units as u
from astropy.time import Time

class TemporaryAP():
    def __init__(self):
        self.data_path = './'
        self.readout_noise = 7.1
        self.gain = 1.91
        self.median_bias = pickle.load(open(self.data_path + 'median_bias.p', 'rb'))
        self.median_bias_error = pickle.load(open(self.data_path + 'median_bias_error.p', 'rb'))
        self.median_normalized_flat = pickle.load(open(self.data_path + 'median_normalized_flat.p', 'rb'))
        self.median_normalized_flat_errors = pickle.load(open(self.data_path + 'median_normalized_flat_errors.p', 'rb'))
        
        self.science_list = np.genfromtxt(self.data_path + '/science/science.list', dtype = str)
        self.science_size = len(self.science_list)
        
        ylen, xlen = np.shape(self.median_bias)
        
        self.X_axis = np.arange(0, xlen, 1)
        self.Y_axis = np.arange(0, ylen, 1)
        
        self.X, self.Y = np.meshgrid(self.X_axis, self.Y_axis)
        
    def aperture_photometry(self):

        self.aperture_flux = np.empty(self.science_size)
        self.sky_bkg = np.empty(self.science_size)

        for ii_science, science_name in enumerate(self.science_list):
            science_fits = fits.open(self.data_path + '/science/' + science_name)

            readout_noise = science_fits[0].header['RDNOISE']
            gain = science_fits[0].header['GAIN']

            science_data = science_fits[0].data + gain
            science_fits.close()

            science_corrected, science_corrected_errors = self.correct_science_frame(science_data, readout_noise)

            x_refined, y_refined = self.compute_centroid(science_corrected, self.x_initial, self.y_initial)
            sky_bkg, sky_bkg_error = self.compute_sky_bkg(science_corrected, x_refined, y_refined)

            science_sky_corrected = science_corrected - sky_bkg
            #compute your own science sky corrected error

            distance = np.sqrt( (self.X - x_refined)**2 + (self.Y - y_refined)**2)
            aperture_selection = (distance < self.aperture_radius)
            aperture_flux = np.sum(science_sky_corrected[aperture_selection])

            self.aperture_flux[ii_science] = aperture_flux
            self.sky_bkg[ii_science] = sky_bkg

                
    def correct_science_frame(self, science_frame, RD_NOISE):

        science_debiased = science_frame - self.median_bias
        science_debiased_errors = np.sqrt(RD_NOISE**2 + science_debiased + self.median_bias_error**2)
        science_corrected = science_debiased / self.median_normalized_flat
        science_corrected_errors = science_corrected * np.sqrt( (science_debiased_errors / science_debiased)**2 + (self.median_normalized_flat_errors / self.median_normalized_flat)**2)

        return science_corrected, science_corrected_errors

    def provide_aperture_parameters(self, sky_inner_radius, sky_outer_radius, aperture_radius, x_initial, y_initial):
        self.sky_inner_radius = sky_inner_radius
        self.sky_outer_radius = sky_outer_radius
        self.aperture_radius = aperture_radius
        self.x_initial = x_initial
        self.y_initial = y_initial

    def compute_sky_bkg(self, science_data, x_pos, y_pos):

        distance = np.sqrt( (self.X - x_pos)**2 + (self.Y - y_pos)**2)
        sky_selection = (distance > self.sky_inner_radius) & (distance < self.sky_outer_radius)
        sky_bkg = np.median(science_data[sky_selection])
        sky_bkg_error = 0 #YOU MUST COMPUTE IT

        return sky_bkg, sky_bkg_error

    def compute_centroid(self, science_data, x_initial, y_initial, maxiterat = 20):

        x_init = x_initial
        y_init = y_initial

        for i in range(0,maxiterat):
            distance = np.sqrt( (self.X - x_init)**2 + (self.Y - y_init)**2)
            selection = (distance < self.sky_inner_radius)

            weighted_x = np.sum(self.X[selection] * science_data[selection])
            weighted_y = np.sum(self.Y[selection] * science_data[selection])
            total_flux = np.sum(science_data[selection])

            x_target_corr = weighted_x/total_flux
            y_target_corr = weighted_y/total_flux

            diffx = np.abs(x_target_corr - x_init)
            diffy = np.abs(y_target_corr - y_init)

            x_init = x_target_corr
            y_init = y_target_corr

            if (diffx < 0.001) and (diffy < 0.001):
                break

        return x_target_corr, y_target_corr