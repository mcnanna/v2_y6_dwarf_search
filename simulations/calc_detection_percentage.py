import argparse
import sys
import os
import glob
import numpy as np
import healpy as hp
import yaml
import astropy.io.fits as fits

import simple.survey
import simple.search
import ugali.utils.projector

import load_data
import simSatellite
import utils


def get_random_loc(survey):
    flist = os.listdir(survey.catalog['dirname'])
    pixels = [int(fname.split('.')[0][-5:]) for fname in flist]
    while True:
        center = np.random.choice(pixels)
        neighbors = hp.get_all_neighbours(survey.catalog['nside'], center, nest=False)
        if all(pixel in pixels for pixel in neighbors):
            break
    
    #corner_vecs = hp.boundaries(survey.catalog['nside'], center, step=1, nest=False)
    #corner_ras, corner_decs = hp.vec2ang(corner_vecs.T, lonlat=True)
    center_ra, center_dec = ugali.utils.healpix.pixToAng(survey.catalog['nside'], center)
    resol = hp.nside2resol(survey.catalog['nside'], arcmin=True)/60.
    while True:
        ra = center_ra + np.random.uniform(-resol/np.sqrt(2.), resol/np.sqrt(2.))
        if ra < 0:
            ra += 360
        dec = center_dec + np.random.uniform(-resol/np.sqrt(2.), resol/np.sqrt(2.))
        if ugali.utils.healpix.angToPix(survey.catalog['nside'], ra, dec) == center:
            break

    return ra, dec


def calc_detection_prob(inputs, survey, abs_mag, a_physical, distance, radec=None, max_trials=100):
    prob = 0
    delta = 1

    sigmas = []

    if abs_mag < -11.99:
        return 1., np.tile(37.5, 20)

    print("{:<3} {:>6},{:<6} {:>5} {:>5}".format('n', 'ra', 'dec', 'sigma', 'prob'))
    while (len(sigmas) < max_trials):
        if (delta < 0.01) and len(sigmas) >= 20: # Percent has stabilized
            if (np.max(sigmas) < 4.5) or (np.min(sigmas) > 10.0): # Prob is sitting at 0 or 100 and it'll never change
                break
            # If neither above condition is met, then prob has stabilized but it still may change

        if radec is not None:
            ra, dec = radec
        else:
            ra, dec = get_random_loc(survey)
        region = simple.survey.Region(survey, ra, dec)

        sim = simSatellite.SimSatellite(inputs, region.ra, region.dec, distance, abs_mag, a_physical)
        data, n_stars = simSatellite.inject(region, sim)
        region.data = data
        sig = simSatellite.search(region, data, mod=ugali.utils.projector.distanceToDistanceModulus(distance))
        sigmas.append(sig)

        old_prob = prob
        prob = np.count_nonzero(np.array(sigmas) > 6)*1./len(sigmas)
        print("{:<3} {:>6.2f},{:<6.2f} {:>5.2f} {:>5.2f}".format(len(sigmas), ra, dec, sig, prob))
        delta = np.abs(prob - old_prob)

    print()
    print("{} total satellites simulated".format(len(sigmas)))
    print("Prob = {}".format(prob))
    print("Average sigma = {}".format(np.mean(sigmas)))
    return prob, sigmas
    
def collect_detection_probs(outname='detection_table'):
    result_files = glob.glob('detection_percentages/*/*.npy')

    out_array = []
    for fname in result_files:
        m, a, d = list(map(float, fname[:-7].split('_')[-3:]))
        sigmas = np.load(fname)
        avg = np.mean(sigmas)
        std = np.std(sigmas)
        prob = np.count_nonzero(sigmas > 6)*1./len(sigmas)
        out_array.append((d, m, a, avg, std, prob))

    dtype = [('distance',float), ('abs_mag',float), ('a_physical',float), ('sigma',float), ('std',float), ('prob',float)]
    fits_out = np.array(out_array, dtype=dtype)
    fits.writeto('detection_percentages/' + outname + '.fits', fits_out)
    
    return fits_out


def calc_density(inputs, abs_mag, a_physical, distance, max_trials=20):
    densities = []

    if abs_mag < -11.99:
       return 99.9, np.tile(99.9, 20)

    while (len(densities) < max_trials):
        ra, dec = 2.87, -38.44 # Shouldn't matter 
        sim = simSatellite.SimSatellite(inputs, ra, dec, distance, abs_mag, a_physical, use_completeness=False, use_efficiency=False)
        stars = sim.stars 
        cut = stars['PSF_MAG_I_CORRECTED'] < 27.0 # Same cutoff Jonah used. Dimmer stars were modeled by a Sersic profile
        stars = stars[cut]

        angseps = ugali.utils.projector.angsep(ra, dec, stars['RA'], stars['DEC'])
    
        radius = sim.a_h
        n = np.count_nonzero(angseps < radius)
        area = np.pi * (radius*3600.)**2 
        density = n/area
        print("{:<3} {:>5.4}".format(len(densities), density))
        densities.append(density)

    print()
    print("{} total satellites simulated".format(len(densities)))
    print("Mean Density = {}".format(np.mean(densities)))
    return densities

def collect_densities(outname='density_table'):
    density_files = glob.glob('stellar_densities/*.npy')

    out_array = []
    for fname in result_files:
        m, a, d = list(map(float, fname[:-7].split('_')[-3:]))
        densities = np.load(fname)
        mean = np.mean(densities)
        std = np.std(densities)
        out_array.append((d, m, a, mean, std))

    dtype = [('distance',float), ('abs_mag',float), ('a_physical',float), ('density',float), ('std',float)]
    fits_out = np.array(out_array, dtype=dtype)
    fits.writeto('stellar_densities/' + outname + '.fits', fits_out)

    return fits_out



def analyze_characteristic_densities(ymlfile, n=100, outlabel=None):
    with open(ymlfile) as f:
        cfg = yaml.safe_load(f)
        survey = simple.survey.Survey(cfg)
    """ 
    cds = []
    for j in range(n):
        ra, dec = get_random_loc(survey)
        region = simple.survey.Region(survey, ra, dec)
        data = region.get_data(type='stars', use_other=True) # type='stars', use_other=True is default
        cd = region.characteristic_density_local(data, 0., 0., None)
        cds.append((ra, dec, cd))
    outname = 'cdls/cdl'
    if outlabel is not None:
        outname += '_' + str(outlabel)
    np.save(outname, cds)
    """

    ra, dec = 3.874, -38.419
    region = simple.survey.Region(survey, ra, dec)
    data = region.get_data(type='stars', use_other=True) # type='stars', use_other=True is default
    cd = region.characteristic_density_local(data, 0., 0., None)
    print("{} is cdl near \\name".format(cd))

    return np.array(cds)



if __name__ == "__main__":
    if '--collect' in sys.argv:
        collect_detection_probs()
        collect_densities()
        sys.exit(0)
    if '--cdl' in sys.argv:
        analyze_characteristic_densities('../des.yaml', n=100, outlabel=sys.argv[-1])
        sys.exit(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--density', action='store_true')
    parser.add_argument('--distance', required=True, type=float, help="kpc")
    parser.add_argument('--abs_mag', required=True, type=float, help="m_v")
    parser.add_argument('--log_a_half', required=True, type=float, help="log(a_physical), in pc")
    parser.add_argument('--max_trials', type=int, default=100)
    parser.add_argument('--ra', type=float)#, default=2.87)
    parser.add_argument('--dec', type=float)#, default=-38.44)
    args = vars(parser.parse_args())
    if (args['ra'] is None) ^ (args['dec'] is None):
        raise ValueError("Either both or neither of --ra and --dec must be specified")

    with open(args['config'], 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
        survey = simple.survey.Survey(cfg)
    inputs = load_data.Inputs(cfg)

    a_physical = 10**args['log_a_half']

    if args['density']: # Only calculate stellar density
        densities = calc_density(inputs, args['abs_mag'], a_physical, args['distance'], max_trials=args['max_trials'])
        np.save('stellar_densities/density_{m:.1f}_{a:.1f}_{d}kpc'.format(m=args['abs_mag'], a=args['log_a_half'], d=int(args['distance'])), densities)
    
    else: # This is the main use of this script 
        if args['--ra'] or args['--dec']:
            radec = ra, dec
        else:
            radec = None
        prob, sigmas = calc_detection_prob(inputs, survey, args['abs_mag'], a_physical, args['distance'], radec=radec, max_trials=args['max_trials'])
        np.save('detection_percentages/{d}/sigs_{m:.1f}_{a:.1f}_{d}kpc'.format(m=args['abs_mag'], a=args['log_a_half'], d=int(args['distance'])), sigmas)


