import predict_satellites
import astropy.io.fits as fits
import ugali.utils.healpix
import skymap
import matplotlib.markers
import matplotlib.pyplot as plt
matplotlib.rcParams['xtick.labelsize'] = 11
matplotlib.rcParams['ytick.labelsize'] = 11
#matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
#plt.ion()
import numpy as np
import plot_utils

footprint = ugali.utils.healpix.read_map('~/Research/y6/v2_y6_dwarf_search/candidate-analysis/data/y6_gold_v2_footprint.fits', nest=True)
des_poly = np.genfromtxt('/Users/mcnanna/Research/y3-mw-sats/data/round19_v0.txt',names=['ra','dec'])

pairs = ('RJ', 'TL', 'RR')
titles = ('Romeo & Juliet', 'Thelma & Louise', 'Romulus & Remus')
#titles = ('Romeo \& Juliet', 'Thelma \& Louise', 'Romulus \& Remus')
psis = (60, 138, 282) # RJ, TL, RR

fig1, axs1 = plt.subplots(3, 1, figsize=(14, 14), num=1)

fig2, axs2 = plt.subplots(2, 2, figsize=(19,9), num=2)
axes2 = axs2[0,0], axs2[0,1], axs2[1,0]

for i, pair in enumerate(pairs):
    psideg = psis[i]
    psi = np.radians(psideg)

    # Load satellits and significances
    sats = predict_satellites.Satellites(pair)
    close_cut = (sats.distance > 300)
    far_cut = (sats.distance < 2000)
    cut = close_cut & far_cut

    sigma_table = fits.open('realizations/{0}/sats_table_{0}_5trials.fits'.format(pair))[1].data
    sat_ras, sat_decs = sats.ra_dec(psi)
    sat_ras, sat_decs = sat_ras[cut], sat_decs[cut]
    sigmas = sigma_table['sig']

    # Determine # in foorprint and # detectable
    pix = ugali.utils.healpix.angToPix(4096, sat_ras, sat_decs, nest=True)
    footprint_cut = (footprint[pix] > 0)
    detectable_cut = (sigmas >= 6.0)
    print('{}: {} detectable'.format(titles[i], np.count_nonzero(detectable_cut & footprint_cut)))


    def custom_scatter(smap,x,y,markers,**kwargs):
        sc = smap.scatter(x,y,**kwargs)
        paths=[]
        for marker in markers:
            marker_obj = matplotlib.markers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
        return sc


    markers = np.tile('o', len(sigmas))
    sizes = np.tile(5.0, len(sigmas))
    
    version = 2 # 1 is original, larger markers in footprint and colored by sigma
    # 2 has larger markers when detetable, colored by distance
    if version == 1:
        sizes[footprint_cut] = 15.0
        color_by = sigmas
        cmap = plot_utils.shiftedColorMap('seismic_r', 2.2, 37.5, 6)
        cbar_label = '$\sigma$'
        vmin, vmax = 2.2, 37.5
    elif version == 2:
        #sizes[detectable_cut] = 20.0 # For circles
        sizes[detectable_cut] = 100.0 # For stars
        markers[detectable_cut] = '*'
        color_by = sats.distance[cut]
        cmap = 'viridis_r'
        cbar_label = 'Distance (kpc)'
        vmin, vmax = 300, 2000

    # Plot v1
    plt.sca(axs1[i])
    smap = skymap.Skymap(projection='mbtfpq', lon_0 = 0)
    custom_scatter(smap, sat_ras, sat_decs, c=color_by, cmap=cmap, vmin=vmin, vmax=vmax, latlon=True, s=sizes, markers=markers, edgecolors='k', linewidths=0.2)
    smap.plot(des_poly['ra'], des_poly['dec'], latlon=True, c='0.25', lw=3, alpha=0.3, zorder=0)
    plt.title(titles[i], family='monospace')

    # Plot v2
    plt.sca(axes2[i])
    smap = skymap.Skymap(projection='mbtfpq', lon_0 = 0)
    custom_scatter(smap, sat_ras, sat_decs, c=color_by, cmap=cmap, vmin=vmin, vmax=vmax, latlon=True, s=sizes, markers=markers, edgecolors='k', linewidths=0.2)
    smap.plot(des_poly['ra'], des_poly['dec'], latlon=True, c='0.25', lw=3, alpha=0.3, zorder=0)
    plt.title(titles[i], family='monospace')

plt.figure(1)
cb = plt.colorbar(ax = axs1, shrink=0.95, aspect=25)
cb.set_label(label=cbar_label, size=14)
plt.savefig('3panel.png', bbox_inches='tight', dpi=200)
plt.close()

plt.figure(2)
plt.subplots_adjust(wspace=0.1)
axs2[1,1].set_axis_off()
cax = fig2.add_axes([0.56, 0.3, 0.3, 0.03]) # left, bottom, width, height
cb = plt.colorbar(cax=cax, orientation='horizontal')
cb.set_label(label=cbar_label, size=13)
plt.savefig('3panel_v2.png', bbox_inches='tight', dpi=200)
plt.close()
