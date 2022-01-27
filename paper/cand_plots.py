"""
Wrapper for $SIMPLE/plotting/diagnostic_plots_paper.py

Makes sure all relevant files are copied over
"""

import argparse
import yaml

import astropy.io.fits as fits
import numpy as np
import healpy as hp
import ugali.utils.healpix

from utils import get_healpixel_files
import simple.survey
import simple.plotting.diagnostic_plots_paper

# Command line arguments
p = argparse.ArgumentParser()
p.add_argument('--config', required=True)
p.add_argument('--infile', required=True)
args = vars(p.parse_args())

# Load config .yaml
with open(args['config'], 'r') as ymlfile:
    cfg = yaml.load(ymlfile)
    #cfg['survey']['fracdet'] = None # Remove fracdet for faster loading while testing
    survey = simple.survey.Survey(cfg)

# Read candidate list, scp necessary files, and plot
candidate_list = fits.open(args['infile'])[1].data
for candidate in candidate_list:
    ra = candidate['RA']
    dec = candidate['DEC']

    center = ugali.utils.healpix.angToPix(32, ra, dec)
    neighbors = hp.get_all_neighbours(32, center)
    pixels = np.concatenate(([center], neighbors))
    get_healpixel_files(cfg, 32, pixels)

    simple.plotting.diagnostic_plots_paper.make_plot(survey, candidate)






