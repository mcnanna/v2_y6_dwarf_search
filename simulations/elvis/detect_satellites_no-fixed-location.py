#!/usr/bin/env python

import argparse
import yaml
import subprocess

import numpy as np
from astropy.table import Table
import astropy.io.fits as fits
from astropy.coordinates import SkyCoord
import ugali.utils.healpix

import matplotlib
matplotlib.use('Agg') # so it works in a screen session
import matplotlib.pyplot as plt
import matplotlib.markers
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
plt.ion()
import skymap
import plot_utils

import simple.survey
import predict_satellites
import sys; sys.path.append('../')
import load_data
import simSatellite

import percent


def count_skymap(pair, survey, inputs, footprint, sats, sat_cut=None, psi=None, n_trials=1):
    subprocess.call('mkdir -p realizations/{}/skymaps_no-fixed-location'.format(pair).split())

    if psi is None and pair=='RJ':
        psi = np.radians(66.)
    if psi is None and pair=='TL':
        psi = np.radians(330.)

    sat_ras, sat_decs = sats.ra_dec(psi)
    if sat_cut is not None:
        sat_ras, sat_decs = sat_ras[sat_cut], sat_decs[sat_cut]
    #sigmas = sigma_table['sig']

    pix = ugali.utils.healpix.angToPix(4096, sat_ras, sat_decs, nest=True)
    footprint_cut = (footprint[pix] > 0)

    sigmas = np.zeros(np.count_nonzero(footprint_cut))
    for i,sat in enumerate(sats[sat_cut][footprint_cut]):
        if sat['M_r'] <= -12: # Too bright, don't bother realizing:
            sigmas[i] = 37.5
            continue

        sigs = []
        for _ in range(n_trials):
            ra, dec = sat_ras[footprint_cut][i], sat_decs[footprint_cut][i]
            region = simple.survey.Region(survey, ra, dec)

            realized_sat = simSatellite.SimSatellite(inputs, ra, dec, sat['distance'], sat['M_r'], sat['r_physical'], ellipticity=0.)
            data, n_stars = simSatellite.inject(region, realized_sat)
            if n_stars==0: # No stars in satellite passing cuts
                sigs.append(2.2) # Approx sig value for 0 stars in this region
                continue
            modulus = 5*np.log10(sat['distance']*1000.)-5
            sig = simSatellite.search(region, data, modulus)
            sigs.append(sig)
            if sig >= 37.5: # Maxing out signficance. Don't need more trials
                break
        sigmas[i] = np.mean(sigs)
        percent.bar(i+1, np.count_nonzero(footprint_cut))

    detectable_cut = (sigmas >= 6.0)
    print("psi = {}: {} total sats, {} detectable".format(np.degrees(psi), np.count_nonzero(footprint_cut), np.count_nonzero(detectable_cut)))

    plt.figure(figsize=(12,8)) 
    smap = skymap.Skymap(projection='mbtfpq', lon_0=0)
    #cmap=plot_utils.shiftedColorMap('seismic_r', min(sigmas), max(sigmas), 6)
    cmap=plot_utils.shiftedColorMap('seismic_r', 2.2, 37.5, 6)

    out_markers = np.tile('o', np.count_nonzero(~footprint_cut))
    out_sizes = np.tile(1.0, np.count_nonzero(~footprint_cut))
    in_markers = np.tile('*', np.count_nonzero(footprint_cut))
    in_sizes = np.tile(50.0, np.count_nonzero(footprint_cut))
    def custom_scatter(smap,x,y,markers,**kwargs):
        sc = smap.scatter(x,y,**kwargs)
        paths=[]
        for marker in markers:
            marker_obj = matplotlib.markers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
        return sc
    custom_scatter(smap, sat_ras[~footprint_cut], sat_decs[~footprint_cut], c='0.4', latlon=True, s=out_sizes, markers=out_markers)
    custom_scatter(smap, sat_ras[footprint_cut], sat_decs[footprint_cut], c=sigmas, cmap=cmap, latlon=True, s=in_sizes, markers=in_markers, edgecolors='k', linewidths=0.2)
    plt.colorbar()
    #Add DES polygon
    des_poly = np.genfromtxt('/afs/hep.wisc.edu/home/mcnanna/data/round19_v0.txt',names=['ra','dec'])
    smap.plot(des_poly['ra'], des_poly['dec'], latlon=True, c='0.25', lw=3, alpha=0.3, zorder=0)
    #Add MW disk
    disk_points = sats.halos.mw_disk()
    transform = sats.halos.rotation(psi)
    dx, dy, dz = np.dot(transform, disk_points.T)
    skycoord = SkyCoord(x=dx, y=dy, z=dz, representation='cartesian').spherical
    dras, ddecs = skycoord.lon.degree, skycoord.lat.degree
    order = np.argsort(dras)
    dras, ddecs = dras[order], ddecs[order]
    smap.plot(dras, ddecs, latlon=True, c='green', lw=3, zorder=0)

    psideg = int(round(np.degrees(psi),0))
    plt.title('$\psi = {}^{{\circ}}$; {} total sats in footprint, {} detectable'.format(psideg, np.count_nonzero(footprint_cut), np.count_nonzero(detectable_cut)))
    #plt.savefig('realizations/{0}/skymaps_no-fixed-location/{0}_skymap_psi={1:0>3d}.png'.format(pair, psideg), bbox_inches='tight', dpi=200)
    plt.savefig('{0}_skymap_psi={1:0>3d}.png'.format(pair, psideg), bbox_inches='tight', dpi=200)
    plt.close()

    return (footprint_cut, detectable_cut)


def rotated_skymaps(pair, survey, inputs, footprint, sats, sat_cut=None, n_trials=1, n_rotations=60):
    subprocess.call('mkdir -p realizations/{}/skymaps_no-fixed-location'.format(pair).split())

    cut_results = []
    for i in range(n_rotations):
        psi = 2*np.pi * float(i)/n_rotations
        cuts = count_skymap(pair, survey, inputs, footprint, sats, sat_cut=sat_cut, psi=psi, n_trials=1)
        cut_results.append(cuts)

    # Save results
    cut_results = np.array(cut_results)
    np.save('realizations/{0}/{0}_skymap_cuts_no-fixed-location'.format(pair),cut_results)

    # Merge skymaps into a .gif
    print('\nCreating .gif...')
    subprocess.call("convert -delay 30 -loop 0 realizations/{0}/skymaps_no-fixed-location/*.png realizations/{0}/{0}_skymap_no-fixed-location.gif".format(pair).split())
    print('Done!')



def summary_plots(pair):
    results = np.load('realizations/{0}/{0}_skymap_cuts_no-fixed-location.npy'.format(pair), allow_pickle=True)

    if args['pair'] == 'RJ':
        title = 'Romeo \& Juliet'
    elif args['pair'] == 'TL':
        title = 'Thelma \& Louise'

    def hist(result, xlabel, outname):
        mx = max(result)
        if mx<=30:
            bins = np.arange(mx+2)-0.5
        else:
            bins = 30
        plt.hist(result, bins=bins)
        if mx <= 20:
            xticks = range(mx+1)
            plt.xticks(xticks)
        plt.xlabel(xlabel)
        plt.title(title)
        plt.savefig('realizations/{}/{}.png'.format(pair, outname), bbox_inches='tight')
        plt.close()

    total_sats = [np.count_nonzero(cuts[0]) for cuts in results]
    hist(total_sats, 'Satellites in footprint', 'total_sats_hist_no-fixed-location')
    detectable_sats = [np.count_nonzero(cuts[1]) for cuts in results]
    hist(detectable_sats, 'Detectable satellites in foorprint', 'detectable_sats_hist_no-fixed-location')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pair', default='RJ')
    p.add_argument('--n_trials', default=1, type=int)
    p.add_argument('--config')
    p.add_argument('--psi', default=None, type=float)
    p.add_argument('-r', '--rotations', action='store_true')
    p.add_argument('-s', '--summary', action='store_true')
    args = vars(p.parse_args())

    if args['psi'] or args['rotations']:
        if args['config'] is None:
            raise ValueError("Config required for significance table creation")
        with open(args['config'], 'r') as ymlfile:
            cfg = yaml.load(ymlfile)
            survey = simple.survey.Survey(cfg)
            inputs = load_data.Inputs(cfg)

        sats = predict_satellites.Satellites(args['pair'])
        # Cut satellites to those outside MW virial radius, but < 2 Mpc distance. 
        close_cut = (sats.distance > 300)
        far_cut = (sats.distance < 2000)
        cut = close_cut & far_cut
        subprocess.call('mkdir -p realizations/{}/skymaps_no-fixed-location'.format(args['pair']).split())
        footprint = ugali.utils.healpix.read_map('~/data/y6_gold_v2_footprint.fits', nest=True)
        #footprint = ugali.utils.healpix.read_map('~/Research/y6/v2_y6_dwarf_search/candidate-analysis/data/y6_gold_v2_footprint.fits', nest=True)
    
    if args['psi'] is not None:
        psi = np.radians(args['psi'])
        count_skymap(args['pair'], survey, inputs, footprint, sats, sat_cut=cut, psi=psi, n_trials=args['n_trials'])
    if args['rotations']:
        rotated_skymaps(args['pair'], survey, inputs, footprint, sats, sat_cut=cut, n_rotations=60, n_trials=args['n_trials'])

    if args['summary']:
        summary_plots(args['pair'])


