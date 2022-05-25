import numpy as np
import healpy as hp
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import matplotlib
plt.ion()

import ugali.utils.healpix

##########

infile = '/Users/mcnanna/Research/y6/v2_y6_dwarf_search/candidate-analysis/data/y6_gold_2_0_hsc_test.fits'
# Could consider using more data, this is a ~2x2.5 deg rectangle overlapping with HSC UDEEP SXDS field
data = fits.open(infile)[1].data

infile_maglim = '/Users/mcnanna/Research/y6/v2_y6_dwarf_search/candidate-analysis/data/y6_gold_2_0_decasu_bdf_nside4096_r_depth.fits'
#maglim_map = hp.read_map(infile_maglim, nest=True)
maglim_map = ugali.utils.healpix.read_map(infile_maglim)

nside = 4096

pix = ugali.utils.healpix.angToPix(nside, data['RA'], data['DEC'], nest=False)

maglim = maglim_map[pix]

#cut = (np.fabs(data_cat['cm_T']) < 0.02) & (data_cat['psf_mag_r'] > 15.) & (data_cat['psf_mag_r'] < 30.)
cut = (data['EXT_MASH'] >= 0) & (data['EXT_MASH'] <= 2) & (data['PSF_MAG_R_CORRECTED'] > 15.) & (data['PSF_MAG_R_CORRECTED'] < 30) & (maglim > -1.6e30)

x = data['PSF_MAG_R_CORRECTED'][cut] - maglim[cut]
y = np.log10(data['PSF_MAG_ERR_R'][cut])

#cut_fit = (x > -3.) & (x < 0.)
cut_fit = (x > 0.) & (x < 1.)

p = np.polyfit(x[cut_fit], y[cut_fit], deg=1)

x_plot = np.linspace(-3., 2., 100)
y_plot = 10**((p[0] * x_plot) + p[1])

mag_bins = np.arange(-6., 4.00001, 0.1)
mag_centers = 0.5 * (mag_bins[0:-1] + mag_bins[1:])
y_centers = np.tile(0., len(mag_centers))
for ii in range(0, len(mag_centers)):
    cut_mag = (x > mag_bins[ii]) & (x < mag_bins[ii + 1])
    y_centers[ii] = np.median(y[cut_mag])
mag_err_centers = 10**y_centers

plt.figure()
plt.yscale('log')
plt.ylim(9*10**-5, 4.5*10**1)
#plt.scatter(data_cat['psf_mag_r'][cut] - maglim[cut], data_cat['psf_mag_err_r'][cut], edgecolor='none', s=1, marker='.', alpha=0.1, label='Data')
plt.scatter(data['PSF_MAG_R_CORRECTED'][cut] - maglim[cut], data['PSF_MAG_ERR_R'][cut], edgecolor='none', s=1, marker='.', alpha=1., label='Data')
plt.plot(x_plot, y_plot, c='red', label='Linear Fit')
plt.scatter(mag_centers, mag_err_centers, c='black', s=10, edgecolor='none', label='Photometric Error Model')
plt.xlabel('Magnitude Relative to Maglim')
plt.ylabel('Photometric Uncertainty (mag)')
plt.legend(loc='upper left', scatterpoints=1, markerscale=4)
plt.savefig('photometric_error_model.png', dpi=150)
plt.close()

# Y6-Y3 comparison
fig, axs = plt.subplots(2, 1, figsize=(6.4, 6.4), sharex=True, gridspec_kw={'height_ratios':[2,1]})
ax1 = axs[0]
ax1.set_yscale('log')
ax1.set_ylim(9*10**-5, 4.5*10**1)
ax1.scatter(data['PSF_MAG_R_CORRECTED'][cut] - maglim[cut], data['PSF_MAG_ERR_R'][cut], edgecolor='none', s=1, marker='.', alpha=1., label='Data')
ax1.plot(x_plot, y_plot, c='red', label='Linear Fit')
ax1.scatter(mag_centers, mag_err_centers, c='black', s=10, edgecolor='none', label='Photometric Error Model')
#ax1.set_xlabel('Magnitude Relative to Maglim')
ax1.set_ylabel('Photometric Uncertainty (mag)')
# Add in Y3 for comparison
d = np.recfromcsv('/Users/mcnanna/Research/y6/v2_y6_dwarf_search/candidate-analysis/data/photo_error_model_y3.csv')
ax1.plot(d['mag'], 10**d['log_mag_err'], linestyle='--', color='0.2', label='Y3')
ax1.legend(loc='upper left', scatterpoints=1, markerscale=4)

ax2 = axs[1]
diff = mag_err_centers - 10**d['log_mag_err']
rel = diff/mag_err_centers
ax2.plot(mag_centers, rel)
ax2.axhline(0, linestyle='solid', color='k', linewidth=0.7)
ax2.set_ylabel('Relative difference')
ax2.set_xlabel('Magnitude Relative to Maglim')

plt.savefig('photometric_error_model_compare.png')
plt.close()


outfile = 'photo_error_model.csv'
writer = open(outfile, 'w')
writer.write('%12s,%12s\n'%('mag', 'log_mag_err'))
for ii in range(0, len(mag_centers)):
    writer.write('%12.3f,%12.3f\n'%(mag_centers[ii], y_centers[ii]))
writer.close()

