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
        dec = center_dec + np.random.uniform(-resol/np.sqrt(2.), resol/np.sqrt(2.))
        if ugali.utils.healpix.angToPix(survey.catalog['nside'], ra, dec) == center:
            break

    return ra, dec


def calc_detection_prob(inputs, survey, abs_mag, a_physical, distance, radec=None, max_trials=100):
    if radec is not None:
        ra, dec = radec
    else:
        ra, dec = get_random_loc(survey)
    region = simple.survey.Region(survey, ra, dec)

    counter = 0
    prob = 0
    delta = 1

    sigmas = []

    if abs_mag < -11.99:
        return 1., np.tile(37.5, 10)

    while (counter < max_trials):
        if (delta < 0.01) and counter >= 20: # Percent has stabilized
            if (np.max(sigmas) < 4.5) or (np.min(sigmas) > 10.0): # Prob is sitting at 0 or 100 and it'll never change
                break
            # If neither above condition is met, then prob is sitting at prob has stabilized but it still may change

        sim = simSatellite.SimSatellite(inputs, region.ra, region.dec, distance, abs_mag, a_physical)
        data, n_stars = simSatellite.inject(region, sim)
        region.data = data
        sig = simSatellite.search(region, data, mod=ugali.utils.projector.distanceToDistanceModulus(distance))
        sigmas.append(sig)

        old_prob = prob
        prob = np.count_nonzero(np.array(sigmas) > 6)*1./len(sigmas)
        print(counter, round(prob,2), round(sig,2))
        delta = np.abs(prob - old_prob)

        counter += 1
    print()
    print("{} total satellites simulated".format(counter))
    print("Prob = {}".format(prob))
    print("Average sigma = {}".format(np.mean(sigmas)))
    return prob, sigmas


def collect_results(outname='detection_table'):
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


if __name__ == "__main__":
    if '--collect' in sys.argv:
        collect_results()
        sys.exit(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
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

    prob, sigmas = calc_detection_prob(inputs, survey, args['abs_mag'], a_physical, args['distance'], max_trials=args['max_trials'])
    np.save('detection_percentages/{d}/sigs_{m:.1f}_{a:.1f}_{d}kpc'.format(m=args['abs_mag'], a=args['log_a_half'], d=int(args['distance'])), sigmas)


