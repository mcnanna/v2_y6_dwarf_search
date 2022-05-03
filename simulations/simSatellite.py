import numpy as np
import healpy as hp
import importlib

import ugali.utils.healpix
import ugali.utils.projector
import ugali.analysis.source
import ugali.analysis.kernel
import ugali.analysis.results

import simple.isochrone
import simple.survey

import utils

class SimSatellite:
    def __init__(self, inputs, lon_centroid, lat_centroid, distance, abs_mag, r_physical, ellipticity=0.):
        # Stolen from ugali/scratch/simulation/simulate_population.py. Look there for a more general function,
        # which uses maglims, extinction, stuff like that
        """
        r_physical is azimuthally averaged half-light radius, pc
        """

        # Probably don't want to parse every time
        s = ugali.analysis.source.Source()

        # Following McConnachie 2012, ellipticity = 1 - (b/a) , where a is semi-major axis and b is semi-minor axis
        r_h = np.degrees(np.arcsin(r_physical/1000. / distance)) # Azimuthally averaged half-light radius
        # See http://iopscience.iop.org/article/10.3847/1538-4357/833/2/167/pdf
        # Based loosely on https://arxiv.org/abs/0805.2945
        position_angle = np.random.uniform(0., 180.) # Random position angle (deg)
        a_h = r_h / np.sqrt(1. - ellipticity) # semi-major axis (deg)
        
            
        # Elliptical kernels take the "extension" as the semi-major axis
        ker = ugali.analysis.kernel.EllipticalPlummer(lon=lon_centroid, lat=lat_centroid, ellipticity=ellipticity, position_angle=position_angle)

        flag_too_extended = False
        if a_h >= 1.0:
            print('Too extended: a_h = %.2f'%(a_h))
            a_h = 1.0
            flag_too_extended = True
            # raise Exception('flag_too_extended')
        ker.setp('extension', value=a_h, bounds=[0.0,1.0])
        s.set_kernel(ker)
        
        distance_modulus = ugali.utils.projector.distanceToDistanceModulus(distance)

        iso_type = type(ugali.isochrone.factory(name='Bressan2012'))
        iso = simple.isochrone.get_isochrone(iso_type, survey='des', age=12., z=0.0001, distance_modulus=distance_modulus)
        s.set_isochrone(iso)
        # Simulate takes stellar mass as an argument, NOT richness
        mag_g, mag_r, mag_i = s.isochrone.simulate(abs_mag) 

        lon, lat = s.kernel.sample_lonlat(len(mag_r))

        # Depth maps
        nside = hp.npix2nside(len(inputs.m_maglim_g)) # Assuming that the maglim maps have same resolution
        pix = ugali.utils.healpix.angToPix(nside, lon, lat)
        maglim_g = inputs.m_maglim_g[pix]
        maglim_r = inputs.m_maglim_r[pix]
        maglim_i = inputs.m_maglim_i[pix]

        # Extintion
        # DES Y3 Gold fiducial
        nside = hp.npix2nside(len(inputs.m_ebv))
        pix = ugali.utils.healpix.angToPix(nside, lon,lat)
        ext = {'g':3.186, 'r':2.140, 'i':1.569}
        mag_extinction_g = ext['g'] * inputs.m_ebv[pix]
        mag_extinction_r = ext['r'] * inputs.m_ebv[pix]
        mag_extinction_i = ext['i'] * inputs.m_ebv[pix]

        
        # Photometric uncertainties are larger in the presence of interstellar dust reddening
        mag_g_error = 0.01 + 10**(inputs.log_photo_error((mag_g + mag_extinction_g) - maglim_g))
        mag_r_error = 0.01 + 10**(inputs.log_photo_error((mag_r + mag_extinction_r) - maglim_r))
        mag_i_error = 0.01 + 10**(inputs.log_photo_error((mag_i + mag_extinction_i) - maglim_i))

        flux_g_meas = utils.magToFlux(mag_g) + np.random.normal(scale=utils.getFluxError(mag_g, mag_g_error))
        mag_g_meas = np.where(flux_g_meas > 0., utils.fluxToMag(flux_g_meas), 99.)
        flux_r_meas = utils.magToFlux(mag_r) + np.random.normal(scale=utils.getFluxError(mag_r, mag_r_error))
        mag_r_meas = np.where(flux_r_meas > 0., utils.fluxToMag(flux_r_meas), 99.)
        flux_i_meas = utils.magToFlux(mag_i) + np.random.normal(scale=utils.getFluxError(mag_i, mag_i_error))
        mag_i_meas = np.where(flux_i_meas > 0., utils.fluxToMag(flux_i_meas), 99.)

        # Includes penalty for interstellar extinction and also include variations in depth
        # Use r band:
        # 24.42 is the median magnitude limit from Y6 according to Keith's slack message
        median_maglim = np.median(inputs.m_maglim_r[ugali.utils.healpix.angToDisc(4096, lon_centroid, lat_centroid, 0.75)])
        cut_detect = (np.random.uniform(size=len(mag_r)) < inputs.completeness(mag_r + mag_extinction_r + (median_maglim - np.clip(maglim_r, 20., 26.))))

        # Absoulte Magnitude
        v = mag_g - 0.487*(mag_g - mag_r) - 0.0249 # Don't know where these numbers come from, copied from ugali
        flux = np.sum(10**(-v/2.5))
        abs_mag_realized = -2.5*np.log10(flux) - distance_modulus

        r_physical = distance * np.tan(np.radians(r_h)) * 1000. # Azimuthally averaged half-light radius, pc
        surface_brightness_realized = ugali.analysis.results.surfaceBrightness(abs_mag_realized, r_physical, distance) # Average within azimuthally averaged half-light radius

        # Turn star info into an array, matching Y6 .fits data format loaded by simple
        dtype = [('RA', '>f8'), ('DEC', '>f8'), ('PSF_MAG_G_CORRECTED', '>f8'), ('PSF_MAG_R_CORRECTED', '>f8'), ('PSF_MAG_I_CORRECTED', '>f8'), ('PSF_MAG_ERR_G', '>f8'), ('PSF_MAG_ERR_R', '>f8'), ('PSF_MAG_ERR_I', '>f8'), ('BDF_T', '>f8')]
        #sat_stars = np.zeros(len(lon), dtype=dtype)
        sat_stars = np.zeros(np.count_nonzero(cut_detect), dtype=dtype)
        sat_stars['RA'] = lon[cut_detect]
        sat_stars['DEC'] = lat[cut_detect]
        sat_stars['PSF_MAG_G_CORRECTED'] = mag_g_meas[cut_detect]
        sat_stars['PSF_MAG_R_CORRECTED'] = mag_r_meas[cut_detect]
        sat_stars['PSF_MAG_I_CORRECTED'] = mag_i_meas[cut_detect]
        sat_stars['PSF_MAG_ERR_G'] = mag_g_error[cut_detect]
        sat_stars['PSF_MAG_ERR_R'] = mag_r_error[cut_detect]
        sat_stars['PSF_MAG_ERR_I'] = mag_i_error[cut_detect]


        self.stars = sat_stars
        self.a_h = a_h
        self.ellipticity = ellipticity
        self.position_angle = position_angle
        self.abs_mag_realized = abs_mag_realized
        self.surface_brightness_realized = surface_brightness_realized
        self.flag_too_extended = flag_too_extended
        self.iso = iso


def inject(region, simSatellite, use_other=True):
    # Cuts
    sel = np.tile(True, len(simSatellite.stars))
    quality = region.survey.catalog['quality']
    cuts = quality.split(' && ')
    for cut in cuts:
        split = cut.split()
        band = split[0]
        limit = float(split[-1])
        sel &= simSatellite.stars[band] < limit

    if use_other:
        for module in region.survey.catalog['other'].split('&&'):
            try: # Python 3
                spec = importlib.util.spec_from_file_location(module, os.getcwd()+'/{}.py'.format(module))
                other = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(other)
            except: # Python 2
                other = importlib.import_module(module.strip())
            sel &= other.sel(region.survey, simSatellite.stars)

    data = region.get_data(use_other=use_other)
    n_stars = np.count_nonzero(sel)
    #print('Injecting {} simulated stars'.format(n_stars))
    data = np.concatenate((data, simSatellite.stars[sel]))

    return data, n_stars


### Right now, the distance is hardcoded as D = 2000 kpc, mod=26.5 ###
def search(region, data, mod=26.5):
# Copied from simple.search
    iso = region.survey.get_isochrone(mod)
    cut = iso.cut_separation(region.survey.band_1.lower(), region.survey.band_2.lower(), data[region.survey.mag_1], data[region.survey.mag_2], data[region.survey.mag_err_1], data[region.survey.mag_err_2], radius=0.1)
    if region.survey.band_3 is not None:
        cut &= iso.cut_separation(region.survey.band_2.lower(), region.survey.band_3.lower(), data[region.survey.mag_2], data[region.survey.mag_3], data[region.survey.mag_err_2], data[region.survey.mag_err_3], radius=0.1)
    data = data[cut]

    region.set_characteristic_density(data)

    x_peak, y_peak = 0.0, 0.0 
    x, y = region.proj.sphereToImage(data[region.survey.catalog['basis_1']], data[region.survey.catalog['basis_2']])
    angsep_peak = np.sqrt((x-x_peak)**2 + (y-y_peak)**2)

    ra_peak, dec_peak, r_peak, sig_peak, distance_modulus, n_obs_peak, n_obs_half_peak, n_model_peak = region.fit_aperture(data, mod, x_peak, y_peak, angsep_peak)
            
    #print('----------------------------------------')
    #print('(ra, dec) = ({:0.2f}, {:0.2f}); {:0.2f} sigma; r = {:0.3f} deg; n_obs = {:n}'.format(ra, dec, sig_peak, r_peak, n_obs_peak))
    return sig_peak

