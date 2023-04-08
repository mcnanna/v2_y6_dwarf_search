import argparse
import numpy as np
import scipy.stats
import healpy as hp
import astropy.io.fits as fits
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.ion()

import ugali.utils.projector
import ugali.utils.healpix


##################################################################

params = {
    'figure.figsize': (8., 6.),
    #'backend': 'eps',
    'axes.labelsize': 18,
    #'text.fontsize': 12,           
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'xtick.major.size': 3,      # major tick size in points
    'xtick.minor.size': 1.5,    # minor tick size in points
    'xtick.major.size': 3,      # major tick size in points
    'xtick.minor.size': 1.5,    # minor tick size in points
    'text.usetex': True,
    #'figure.figsize': fig_size,
    'font.family':'serif',
    'font.serif':'Computer Modern Roman Bold',
    'font.size': 18
    }
matplotlib.rcParams.update(params)

#matplotlib.style.use('des_dr1')

##################################################################


p = argparse.ArgumentParser()
p.add_argument('--merge', help='Use merged HSC+SPLASH classifier as truth', action='store_true')
p.add_argument('--ext', help='DES stars have EXT_MASH <= ext', type=int, default=2)
args = vars(p.parse_args())

# HSC

if args['merge']:
    infile_hsc = '/Users/mcnanna/Research/y6/v2_y6_dwarf_search/candidate-analysis/data/merge_hsc_splash.fits'
    print 'Reading %s ...'%(infile_hsc)
    d_hsc = fits.open(infile_hsc)[1].data
else:
    infiles_hsc = ['/Users/mcnanna/Research/y6/v2_y6_dwarf_search/candidate-analysis/data/hsc_dr1_udeep_sxds_gri_test.fits.gz',
                   '/Users/mcnanna/Research/y6/v2_y6_dwarf_search/candidate-analysis/data/hsc_dr1_deep_deep2_3_gri_test.fits.gz',
                   '/Users/mcnanna/Research/y6/v2_y6_dwarf_search/candidate-analysis/data/hsc_dr1_wide_vvds_gri_test.fits.gz']
    fields_hsc = ['sxds', 'deep2_3', 'vvds']

    d_hsc = []
    f_hsc = []
    for infile_hsc, field_hsc in zip(infiles_hsc, fields_hsc):
        print 'Reading %s ...'%(infile_hsc)
        reader_hsc = fits.open(infile_hsc)
        d_hsc.append(reader_hsc[1].data)
        reader_hsc.close()
        f_hsc.append(np.tile(field_hsc, len(reader_hsc[1].data)))
    d_hsc = np.concatenate(d_hsc)
    f_hsc = np.concatenate(f_hsc)

#cut = ~np.isnan(d_hsc['imag_psf']) & ~np.isnan(d_hsc['icmodel_mag']) & (d_hsc['imag_psf'] < 25.5)
#cut = ~np.isnan(d_hsc['imag_psf']) & ~np.isnan(d_hsc['icmodel_mag']) & (d_hsc['icmodel_mag'] < 27.)
cut = ~np.isnan(d_hsc['imag_psf']) & ~np.isnan(d_hsc['icmodel_mag']) & ~np.isnan(d_hsc['rmag_psf']) & ~np.isnan(d_hsc['rcmodel_mag']) 
cut = cut & ~np.isinf(d_hsc['imag_psf']) & ~np.isinf(d_hsc['icmodel_mag']) & ~np.isinf(d_hsc['rmag_psf']) & ~np.isinf(d_hsc['rcmodel_mag']) 
cut = cut & (d_hsc['rcmodel_mag'] < 27.)
d_hsc = d_hsc[cut]
#f_hsc = f_hsc[cut]
cut_spatial = (d_hsc['ra'] < 180.) | (np.fabs(d_hsc['dec']) < 2.)
d_hsc = d_hsc[cut_spatial]
#f_hsc = f_hsc[cut_spatial]

# DES

infile_des = '/Users/mcnanna/Research/y6/v2_y6_dwarf_search/candidate-analysis/data/y6_gold_2_0_hsc_test.fits' # Includes gri photometry
print 'Reading %s ...'%(infile_des)
d_des = fits.open(infile_des)[1].data
cut_spatial = (d_des['RA'] < 180.) | (np.fabs(d_des['DEC']) < 2.)
d_des = d_des[cut_spatial]

cut_clean = (d_des['IMAFLAGS_ISO_G'] == 0) & (d_des['IMAFLAGS_ISO_R'] == 0) & (d_des['IMAFLAGS_ISO_I'] == 0) \
            & (d_des['FLAGS_GOLD'] == 0)
d_des = d_des[cut_clean]

##################################################################

NSIDE = 4096

# Fracdet DES

infile_fracdet_des = '/Users/mcnanna/Research/y6/v2_y6_dwarf_search/candidate-analysis/data/y6_gold_v2_footprint.fits'
m_fracdet_des = hp.read_map(infile_fracdet_des, nest=True)

# HSC mask

if args['merge']:
    pix_hsc_mask = np.load('/Users/mcnanna/Research/y6/v2_y6_dwarf_search/candidate-analysis/data/merge_mask_pix_4096_nest_cel.npy')
else:
    pix_hsc_mask = np.load('/Users/mcnanna/Research/y6/v2_y6_dwarf_search/candidate-analysis/data/hsc_mask_pix_4096_nest_cel.npy')
m_hsc_mask = np.tile(hp.UNSEEN, hp.nside2npix(NSIDE))
m_hsc_mask[pix_hsc_mask] = 1.

pix_joint_mask = np.intersect1d(pix_hsc_mask, np.nonzero(m_fracdet_des > 0.5)[0]) 

print "{} degrees^2 covered".format(len(pix_joint_mask) * hp.nside2pixarea(NSIDE, degrees=True))

##################################################################

# Apply joint mask

pix_des = ugali.utils.healpix.angToPix(NSIDE, d_des['RA'], d_des['DEC'], nest=True)
cut_des = np.in1d(pix_des, pix_joint_mask)
d_des = d_des[cut_des]

pix_hsc = ugali.utils.healpix.angToPix(NSIDE, d_hsc['ra'], d_hsc['dec'], nest=True)
cut_hsc = np.in1d(pix_hsc, pix_joint_mask)
d_hsc = d_hsc[cut_hsc]

##################################################################
if args['merge']:
    cut_star_hsc = (d_hsc['ext_combo'] == 0)
    cut_gal_hsc = ~cut_star_hsc
else:
    hsc_concentration_i = (d_hsc['imag_psf'] - d_hsc['icmodel_mag'])
    #cut_star_hsc_temp = (hsc_concentration_i < 0.03) | ((hsc_concentration_i < 0.05) & (d_hsc['imag_psf'] < 23.)) # Original
    cut_star_hsc_temp = (hsc_concentration_i < 0.03) | ((hsc_concentration_i < 0.1) & (d_hsc['imag_psf'] < 22.)) # Updated for DR1
    cut_star_hsc = cut_star_hsc_temp & (d_hsc['imag_psf'] < 27.0)
    cut_gal_hsc = ~cut_star_hsc_temp & (d_hsc['imag_psf'] < 27.0)
d_hsc_star = d_hsc[cut_star_hsc]

bins = np.arange(18., 27., 0.1)
plt.figure()
plt.yscale('log')
plt.hist(d_hsc['rmag_psf'][cut_star_hsc], bins=bins, color='red', histtype='step', label='Stars')
plt.hist(d_hsc['rmag_psf'][cut_gal_hsc], bins=bins, color='blue', histtype='step', label='Galaxies')
plt.legend(loc='upper left')
plt.xlabel('r (HSC)')
plt.title('Object counts ({})'.format('HSC+SPLASH' if args['merge'] else 'HSC'))
plt.savefig('completeness_plots/{}_hist.png'.format('merge' if args['merge'] else 'hsc'))

##################################################################

# For stars
#A_r_1 = -0.0610454250981
#A_r_0 = 0.0141346761502
A_r_1 = -0.133724
A_r_0 = 0.003088
r_hsc_converted = (A_r_1 * (d_hsc_star['rcmodel_mag'] - d_hsc_star['icmodel_mag'])) + d_hsc_star['rcmodel_mag'] + A_r_0
A_g_1 = -0.016887
A_g_0 = 0.029986
g_hsc_converted = (A_g_1 * (d_hsc_star['gcmodel_mag'] - d_hsc_star['rcmodel_mag'])) + d_hsc_star['gcmodel_mag'] + A_g_0

# Deredden
r_hsc_converted = r_hsc_converted - d_hsc_star['a_r']
g_hsc_converted = g_hsc_converted - d_hsc_star['a_g']

##################################################################

# Star classification for DES
"""
selection_1 = (d_des['CM_T'] + 5. * d_des['CM_T_err']) > 0.1
selection_2 = (d_des['CM_T'] + 1. * d_des['CM_T_err']) > 0.05
selection_3 = (d_des['CM_T'] - 1. * d_des['CM_T_err']) > 0.02 # 0.02
ext = selection_1.astype(int) + selection_2.astype(int) + selection_3.astype(int)
"""
ext = d_des['EXT_MASH']

cut_classified_star = (ext <= args['ext']) & (ext >= 0)

##################################################################

print 'Matching catalogs ...'

# DES any and HSC star

match_des, match_hsc, angsep = ugali.utils.projector.match(d_des['RA'], d_des['DEC'], d_hsc_star['ra'], d_hsc_star['dec'], tol=1/3600.)

cut_match_des = np.tile(False, len(d_des))
cut_match_des[match_des] = True # DES objects that have a match with an HSC star

cut_match_hsc = np.tile(False, len(d_hsc_star))
cut_match_hsc[match_hsc] = True # HSC stars that have a match with a DES object

# DES star and HSC star

match_des_classified_star, match_hsc_classified_star, angsep = ugali.utils.projector.match(d_des['RA'][cut_classified_star], d_des['DEC'][cut_classified_star], 
                                                                                           d_hsc_star['ra'], d_hsc_star['dec'], 
                                                                                           tol=1/3600.)

cut_match_des_classified_star = np.tile(False, len(d_des[cut_classified_star]))
cut_match_des_classified_star[match_des_classified_star] = True # DES stars that have a match with an HSC star

cut_match_hsc_classified_star = np.tile(False, len(d_hsc_star))
cut_match_hsc_classified_star[match_hsc_classified_star] = True # HSC stars that have a match with a DES star

mag_bins = np.arange(20., 27.00001, 0.1)
mag_centers = 0.5 * (mag_bins[0:-1] + mag_bins[1:])

efficiency = np.tile(0., len(mag_centers))
efficiency_star = np.tile(0., len(mag_centers))
for ii in range(0, len(mag_bins) - 1):
    cut_mag = (g_hsc_converted > mag_bins[ii]) & (g_hsc_converted < mag_bins[ii + 1])
    cut_mag_match = (g_hsc_converted > mag_bins[ii]) & (g_hsc_converted < mag_bins[ii + 1]) & cut_match_hsc
    cut_mag_match_classified_star = (g_hsc_converted > mag_bins[ii]) & (g_hsc_converted < mag_bins[ii + 1]) & cut_match_hsc_classified_star
    efficiency[ii] = (1. * np.sum(cut_mag_match)) / np.sum(cut_mag)
    efficiency_star[ii] = (1. * np.sum(cut_mag_match_classified_star)) / np.sum(cut_mag)

##################################################################

# DES star and HSC star
"""
match_des_classified_star, match_hsc_classified_star, angsep = ugali.utils.projector.match(d_des['RA'][cut_classified_star], d_des['DEC'][cut_classified_star], 
                                                                                           d_hsc_star['ra'], d_hsc_star['dec'], 
                                                                                           tol=1/3600.)

cut_match_des_classified_star = np.tile(False, len(d_des))
cut_match_des_classified_star[match_des_classified_star] = True # DES stars that have a match with an HSC star

cut_match_hsc_classified_star = np.tile(False, len(d_hsc_star))
cut_match_hsc_classified_star[match_hsc_classified_star] = True # HSC stars that have a match with a DES star
"""
"""
contamination = np.tile(0., len(mag_centers))
contamination_star = np.tile(0., len(mag_centers))
for ii in range(0, len(mag_bins) - 1):
    cut_mag = (d_des['psf_mag_r_corrected'] > mag_bins[ii]) & (d_des['psf_mag_r_corrected'] < mag_bins[ii + 1])  
    print '~~~~~ %.2f < r < %.2f ~~~~~'%(mag_bins[ii], mag_bins[ii + 1])
    print np.sum(cut_mag & cut_classified_star & ~cut_match_des_classified_star)
    print np.sum(cut_mag & cut_classified_star)
    print np.sum(cut_mag & cut_classified_star & ~cut_match_des_classified_star)
    print np.sum(cut_mag & cut_classified_star & cut_match_des)                                
    contamination[ii] = (1. * np.sum(cut_mag & cut_classified_star & ~cut_match_des_classified_star)) / np.sum(cut_mag & cut_classified_star)
    contamination_star[ii] = (1. * np.sum(cut_mag & cut_classified_star & ~cut_match_des_classified_star)) / np.sum(cut_mag & cut_classified_star & cut_match_des)
"""
# DES star and HSC any

match_des_classified_star, match_hsc_classified_star, angsep = ugali.utils.projector.match(d_des['RA'][cut_classified_star], d_des['DEC'][cut_classified_star], 
                                                                                           d_hsc['ra'], d_hsc['dec'], 
                                                                                           tol=1/3600.)

contamination = np.tile(0., len(mag_centers))
for ii in range(0, len(mag_bins) - 1):
    cut_mag = (d_des['psf_mag_r_corrected'][cut_classified_star][match_des_classified_star] > mag_bins[ii]) \
              & (d_des['psf_mag_r_corrected'][cut_classified_star][match_des_classified_star] < mag_bins[ii + 1])
    contamination[ii] = (1. * np.sum(cut_mag & cut_gal_hsc[match_hsc_classified_star])) / np.sum(cut_mag)

"""
contamination = np.tile(0., len(mag_centers))
for ii in range(0, len(mag_bins) - 1):
    cut_mag = (d_des['psf_mag_r_corrected'][cut_classified_star] > mag_bins[ii]) \
              & (d_des['psf_mag_r_corrected'][cut_classified_star] < mag_bins[ii + 1])
    contamination[ii] = (1. * np.sum(cut_mag & ~cut_match_des_classified_star)) / np.sum(cut_mag)
"""
##################################################################

plt.figure()
plt.axvspan(25.0, 27.0, color='red', alpha=0.2)
plt.plot(mag_centers, efficiency, c='black', ls='--', lw=2, label='All Detections')
plt.plot(mag_centers, efficiency_star, c='black', ls='-', lw=2, label='Stellar Sample')
plt.legend(loc='center left', frameon=True)
plt.xlabel('r (DES)')
plt.ylabel('Stellar Selection Efficiency')
plt.title('Stellar Classification ({}{})'.format('HSC+SPLASH' if args['merge'] else 'HSC', r', EXT$\leq {}$'.format(args['ext'])))
plt.xlim(20., 27.)
plt.ylim(0., 1.)
ax2 = plt.gca().twinx()
ax2.plot(mag_centers, contamination, c='red', lw=2)
ax2.set_ylabel('Galaxy Contamination', color='red')
ax2.tick_params('y', colors='red')
plt.ylim(0., 1.)
#plt.plot(mag_centers, contamination_star, c='orange')
outname = 'completeness_plots/y6_stellar_classification_summary_r_ext{}{}.png'.format(args['ext'], '_merge' if args['merge'] else '')
plt.savefig(outname, dpi=200)

##################################################################

outfile = 'y6_gold_v2_stellar_classification_summary_r_ext{}{}.csv'.format(args['ext'], '_merge' if args['merge'] else '')
writer = open(outfile, 'w')
writer.write('%12s,%12s,%12s\n'%('mag_r', 'eff', 'eff_star'))
for ii in range(0, len(mag_centers)):
    writer.write('%12.3f,%12.3f,%12.3f\n'%(mag_centers[ii], efficiency[ii], efficiency_star[ii]))
writer.close()

##################################################################


