"""
This script combines information from HSC DR1 and SPLASH

HSC DR1: https://arxiv.org/abs/1702.08449

SPLASH: https://arxiv.org/abs/1711.05280

Details at https://cdcvs.fnal.gov/redmine/projects/des-y6/wiki/Y6A1_Object_Classification_HSC-SSP_DR2
"""

import numpy as np
import fitsio
import matplotlib.pyplot as plt

import ugali.utils.projector

plt.ion()

##########

save = True

infile_hsc = '/Users/mcnanna/Research/y6/v2_y6_dwarf_search/candidate-analysis/data/hsc_dr1_udeep_sxds_gri_test.fits'
d_hsc = fitsio.read(infile_hsc)

infile_splash = '/Users/mcnanna/Research/y6/v2_y6_dwarf_search/candidate-analysis/data/SPLASH_SXDF_Mehta+_v1.6.fits'
columns = ['ra', 'dec', 
           'A', 'B',
           'STAR_FLAG',
           'MAG_APER_hsc_g', 'MAG_APER_hsc_r', 'MAG_APER_hsc_i', 'MAG_APER_hsc_z',
           'MAG_APER_video_j', 'MAG_APER_video_h', 'MAG_APER_video_ks']
d_splash = fitsio.read(infile_splash, columns=columns)

##########

cut_hsc = np.isfinite(d_hsc['icmodel_mag']) & ~np.isnan(d_hsc['icmodel_mag']) & \
           np.isfinite(d_hsc['imag_psf']) & ~np.isnan(d_hsc['imag_psf'])
d_hsc = d_hsc[cut_hsc]

cut_splash = (d_splash['MAG_APER_hsc_g'][:,1] > 0.) & \
             (d_splash['MAG_APER_hsc_i'][:,1] > 0.) & \
             (d_splash['MAG_APER_hsc_z'][:,1] > 0.) & \
             (d_splash['MAG_APER_video_ks'][:,1] > 0.)
d_splash = d_splash[cut_splash]

##########

match_hsc, match_splash, angsep = ugali.utils.projector.match(d_hsc['ra'], d_hsc['dec'], 
                                                              d_splash['RA'], d_splash['DEC'],
                                                              tol=1/3600.)
d_hsc_match = d_hsc[match_hsc]
d_splash_match = d_splash[match_splash]

# Check distribution of fluxes before and after matching
bins = np.arange(14., 30., 0.2)
plt.figure()
plt.yscale('log')
plt.hist(d_hsc['icmodel_mag'], bins=bins,
           histtype='step', color='blue', label='HSC')
plt.hist(d_hsc_match['icmodel_mag'], bins=bins,
           histtype='step', color='blue', lw=2, label='HSC (matched)')
plt.hist(d_splash['MAG_APER_hsc_i'][:,1], bins=bins,
           histtype='step', color='black', label='SPLASH APER2')
plt.hist(d_splash_match['MAG_APER_hsc_i'][:,1], bins=bins,
           histtype='step', color='black', lw=2, label='SPLASH (matched)')
plt.legend(loc='upper left')

# Check astrometry and matching
bins = np.linspace(-1., 1., 101)
plt.figure()
plt.hist(3600. * (d_hsc_match['ra'] - d_splash_match['RA']), bins=bins, histtype='step')
plt.hist(3600. * (d_hsc_match['dec'] - d_splash_match['DEC']), bins=bins, histtype='step')

# Looks like we're good to use the match
d_hsc = d_hsc_match
d_splash = d_splash_match

##########

gz = d_splash['MAG_APER_hsc_g'][:,1] - d_splash['MAG_APER_hsc_z'][:,1]
zk = d_splash['MAG_APER_hsc_z'][:,1] - d_splash['MAG_APER_video_ks'][:,1]
cut_star_color = (zk < (0.5 * gz) - 0.5)
cut_gal_color = ~cut_star_color

cut_star_morph = ((d_hsc['imag_psf'] - d_hsc['icmodel_mag']) < 0.015)
cut_gal_morph = ~cut_star_morph

cut_star = (cut_star_morph & cut_star_color)
cut_gal = ~cut_star

cut_star_lephare = (d_splash['STAR_FLAG'] == 1)
cut_gal_lephare = (d_splash['STAR_FLAG'] == 0)
gz = np.linspace(-1., 5., 100)
zk = (0.5 * gz) - 0.5
for mag_min, mag_max in [[23., 24.],
                         [24., 25.]]:
    #mag_min, mag_max = 23., 24.
    #cut_mag = (d_splash['MAG_APER_hsc_i'][:,1] > mag_min) & (d_splash['MAG_APER_hsc_i'][:,1] < mag_max)
    cut_mag = (d_hsc['imag_psf'] > mag_min) & (d_hsc['imag_psf'] < mag_max)
    bins = np.linspace(-1., 5., 150)
    plt.figure()
    plt.hist2d((d_splash['MAG_APER_hsc_g'][:,1] - d_splash['MAG_APER_hsc_z'][:,1])[cut_mag],
                 (d_splash['MAG_APER_hsc_z'][:,1] - d_splash['MAG_APER_video_ks'][:,1])[cut_mag],
                 bins=(bins, bins), cmin=1, cmap='binary')
    plt.colorbar(label='Density of All Objects')
    #plt.scatter((d_splash['MAG_APER_hsc_g'][:,1] - d_splash['MAG_APER_hsc_z'][:,1])[cut_star_lephare],
    #              (d_splash['MAG_APER_hsc_z'][:,1] - d_splash['MAG_APER_video_ks'][:,1])[cut_star_lephare],
    #              s=1, edgecolors='none', c='red', label='Stellar Locus')
    #plt.scatter((d_splash['MAG_APER_hsc_g'][:,1] - d_splash['MAG_APER_hsc_z'][:,1])[cut_mag & cut_gal_color & cut_star_morph],
    #              (d_splash['MAG_APER_hsc_z'][:,1] - d_splash['MAG_APER_video_ks'][:,1])[cut_mag & cut_gal_color & cut_star_morph],
    #              s=1, edgecolors='none', c='blue', label='Unresolved w/ Galaxy Colors')
    plt.scatter((d_splash['MAG_APER_hsc_g'][:,1] - d_splash['MAG_APER_hsc_z'][:,1])[cut_mag & cut_star_morph],
                  (d_splash['MAG_APER_hsc_z'][:,1] - d_splash['MAG_APER_video_ks'][:,1])[cut_mag & cut_star_morph],
                  s=1, edgecolors='none', c='red', label='Stars: Morphology')
    plt.plot(gz, zk, c='blue', lw=1, ls='--')
    plt.xlim(-1., 5.)
    plt.ylim(-1., 3.)
    plt.xlabel('g - z')
    plt.ylabel('z - Ks')
    #plt.title('%.1f < MAG_APER_hsc_i < %.1f'%(mag_min, mag_max))
    plt.title('%.1f < imag_psf < %.1f'%(mag_min, mag_max))
    plt.legend(loc='upper right', markerscale=5)
    if save:
        plt.savefig('merge_plots/merge_hsc_splash_gz-zk_%.1f-%.1f.png'%(mag_min, mag_max))

for mag_min, mag_max in [[23., 24.],
                         [24., 25.]]:
    cut_mag = (d_hsc['imag_psf'] > mag_min) & (d_hsc['imag_psf'] < mag_max)
    bins = np.linspace(-0.1, 1., 201)
    plt.figure()
    plt.hist((d_hsc['imag_psf'] - d_hsc['icmodel_mag'])[cut_mag & cut_star_color],
               bins=bins, histtype='step', color='red', label='Stars: Color')
    plt.hist((d_hsc['imag_psf'] - d_hsc['icmodel_mag'])[cut_mag & cut_gal_color],
               bins=bins, histtype='step', color='black', label='Galaxies: Color')
    plt.axvline(0.015, c='blue', ls='--')
    plt.xlabel('imag_psf - icmodel_mag')
    plt.ylabel('Counts')
    plt.title('%.1f < MAG_APER_hsc_i < %.1f'%(mag_min, mag_max))
    plt.legend(loc='upper right')
    plt.xlim(-0.1, 1.0)
    if save:
        plt.savefig('merge_plots/merge_hsc_splash_hist_psf-cmodel_%.1f-%.1f.png'%(mag_min, mag_max))

#cut_star_morph = ((d_hsc['imag_psf'] - d_hsc['icmodel_mag']) < 0.015)
# Where do the compact "galaxies" live in color space?
#cut_star = (cut_star_morph & cut_star_color)
#cut_gal = ~cut_star

bins = np.arange(16., 30., 0.1)
plt.figure()
plt.yscale('log')
plt.hist(d_hsc['imag_psf'][cut_star_morph], 
           bins=bins, histtype='step', color='red', label='Stars: Morphology')
plt.hist(d_hsc['imag_psf'][cut_star_color], 
           bins=bins, histtype='step', lw=2, alpha=0.5, color='red', label='Stars: Color')
plt.hist(d_hsc['imag_psf'][cut_star], 
           bins=bins, histtype='step', lw=2, color='red', label='Stars: Combined')
plt.hist(d_hsc['imag_psf'][cut_gal], 
           bins=bins, histtype='step', lw=2, color='black', label='Galaxies')
plt.legend(loc='upper left')
plt.xlabel('imag_psf')
plt.ylabel('Counts')
plt.xlim(18., 30.)
plt.ylim(10., plt.ylim()[1])
if save:
    plt.savefig('merge_plots/merge_hsc_splash_hist_i.png')

for mag_min, mag_max in [[23., 24.],
                         [24., 25.]]:
    #mag_min, mag_max = 24.5, 25.5
    cut_mag = (d_hsc['imag_psf'] > mag_min) & (d_hsc['imag_psf'] < mag_max)
    plt.figure()
    plt.scatter((d_hsc['gmag_psf'] - d_hsc['rmag_psf'])[cut_mag & cut_gal],
                  (d_hsc['rmag_psf'] - d_hsc['imag_psf'])[cut_mag & cut_gal],
                  s=1, edgecolors='none', c='0.5', label='Galaxies')
    plt.scatter((d_hsc['gmag_psf'] - d_hsc['rmag_psf'])[cut_mag & cut_star_morph],
                  (d_hsc['rmag_psf'] - d_hsc['imag_psf'])[cut_mag & cut_star_morph],
                  s=1, edgecolors='none', c='red', label='Stars: Morphology')
    plt.scatter((d_hsc['gmag_psf'] - d_hsc['rmag_psf'])[cut_mag & cut_star],
                  (d_hsc['rmag_psf'] - d_hsc['imag_psf'])[cut_mag & cut_star],
                  s=1, edgecolors='none', c='blue', label='Stars: Combined')
    plt.xlim(-0.5, 2.5)
    plt.ylim(-0.5, 2.5)
    plt.xlabel('g - r')
    plt.ylabel('r - i')
    plt.legend(loc='upper right', markerscale=5)
    plt.title('%.1f < imag_psf < %.1f'%(mag_max, mag_min))
    if save:
        plt.savefig('merge_plots/merge_hsc_splash_gr-ri_%.1f-%.1f.png'%(mag_min, mag_max))

plt.figure()
plt.scatter((d_hsc['gmag_psf'] - d_hsc['rmag_psf'])[cut_gal],
              (d_hsc['gmag_psf'])[cut_gal],
              s=1, edgecolors='none', c='0.5', label='Galaxies')
plt.scatter((d_hsc['gmag_psf'] - d_hsc['rmag_psf'])[cut_star_morph],
              (d_hsc['gmag_psf'])[cut_star_morph],
              s=1, edgecolors='none', c='red', label='Stars: Morphology')
plt.scatter((d_hsc['gmag_psf'] - d_hsc['rmag_psf'])[cut_star],
              (d_hsc['gmag_psf'])[cut_star],
              s=1, edgecolors='none', c='blue', label='Stars: Combined')
plt.xlim(-0.5, 2.0)
plt.ylim(27., 19.)
plt.xlabel('g - r')
plt.ylabel('g')
plt.legend(loc='upper right', markerscale=5)
if save:
    plt.savefig('merge_plots/merge_hsc_splash_gr-g.png')

plt.figure()
plt.scatter(d_hsc['imag_psf'][cut_gal],
              (d_hsc['imag_psf'] - d_hsc['icmodel_mag'])[cut_gal],
              s=1, edgecolors='none', c='0.5', label='Galaxies: Combined')
plt.scatter(d_hsc['imag_psf'][cut_star],
              (d_hsc['imag_psf'] - d_hsc['icmodel_mag'])[cut_star],
              s=1, edgecolors='none', c='red', label='Stars: Combined')
plt.xlim(18., 27.)
plt.ylim(-0.03, 0.1)
plt.xlabel('imag_psf')
plt.ylabel('imag_psf - icmodel_mag')
plt.legend(loc='upper left', markerscale=5)
if save:
    plt.savefig('merge_plots/merge_hsc_splash_i_psf-cmodel.png')

# Create a merged data product
outfile = '/Users/mcnanna/Research/y6/v2_y6_dwarf_search/candidate-analysis/data/merge_hsc_splash.fits'
f_out = fitsio.FITS(outfile,'rw')

d_out = {'ra': d_hsc_match['ra'],
         'dec': d_hsc_match['dec'],
         'ext_color': cut_gal_color.astype(int),
         'ext_morph': cut_gal_morph.astype(int),
         'ext_combo': cut_gal.astype(int),
         'gmag_psf': d_hsc['gmag_psf'],
         'rmag_psf': d_hsc['rmag_psf'],
         'imag_psf': d_hsc['imag_psf'],
         'gcmodel_mag': d_hsc['gcmodel_mag'],
         'rcmodel_mag': d_hsc['rcmodel_mag'],
         'icmodel_mag': d_hsc['icmodel_mag'],
         'a_g': d_hsc['a_g'],
         'a_r': d_hsc['a_r'],
         'a_i': d_hsc['a_i']}
if save:
    f_out.write(d_out)
