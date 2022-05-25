import numpy as np
import astropy.io.fits as fits
import healpy as hp
import ugali.utils.healpix
import ugali.utils.projector
import argparse
import matplotlib.pyplot as plt
plt.ion()

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


##################################################################

# Star classification for DES
ext = d_des['EXT_MASH']

cut_classified_star = (ext <= args['ext']) & (ext >= 0)

##################################################################


# Matching
# DES star and HSC star

match_des_classified_star, match_hsc_classified_star, angsep = ugali.utils.projector.match(d_des['RA'][cut_classified_star], d_des['DEC'][cut_classified_star], 
                                                                                           d_hsc_star['ra'], d_hsc_star['dec'], 
                                                                                           tol=1/3600.)

cut_match_des_classified_star = np.tile(False, len(d_des[cut_classified_star]))
cut_match_des_classified_star[match_des_classified_star] = True # DES stars that have a match with an HSC star

cut_match_hsc_classified_star = np.tile(False, len(d_hsc_star))
cut_match_hsc_classified_star[match_hsc_classified_star] = True # HSC stars that have a match with a DES star


# Linear color-space fit
# De-redden HSC mags
g_hsc_corrected = d_hsc_star['gcmodel_mag'] - d_hsc_star['a_g']
r_hsc_corrected = d_hsc_star['rcmodel_mag'] - d_hsc_star['a_r']
i_hsc_corrected = d_hsc_star['icmodel_mag'] - d_hsc_star['a_i']

xr = r_hsc_corrected[cut_match_hsc_classified_star] - i_hsc_corrected[cut_match_hsc_classified_star]
yr = d_des['PSF_MAG_R_CORRECTED'][cut_classified_star][cut_match_des_classified_star] - r_hsc_corrected[cut_match_hsc_classified_star]
A_r_1, A_r_0 = np.polyfit(xr, yr, deg=1)
print(A_r_0, A_r_1)

plt.figure()
plt.hist(d_des['PSF_MAG_R_CORRECTED'][cut_classified_star][cut_match_des_classified_star]-r_hsc_corrected[cut_match_hsc_classified_star], bins=100)

plt.figure()
plt.scatter(d_des['PSF_MAG_R_CORRECTED'][cut_classified_star][cut_match_des_classified_star], r_hsc_corrected[cut_match_hsc_classified_star], s=1)

plt.figure()
plt.scatter(xr, yr, s=1)
plt.plot(xr, map(lambda x: A_r_1*x + A_r_0, xr), c='red')



