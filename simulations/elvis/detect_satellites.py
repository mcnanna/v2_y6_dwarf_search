#!/usr/bin/env python

import argparse
import yaml
import subprocess

import numpy as np
from astropy.table import Table
import astropy.io.fits as fits
from astropy.coordinates import SkyCoord
import ugali.utils.healpix

import skymap
import plot_utils
import matplotlib.pyplot as plt
import matplotlib.markers
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
plt.ion()
import plot_utils

import simple.survey
import predict_satellites
import sys; sys.path.append('../') # Allows access to the below "more general" function used for more than just ELVIS
import load_data
import simSatellite

import percent

def create_sigma_table(sats, inputs, survey, outname=None, n_trials=1):
    """
    Calculates detection significance for satellites occupying subhalos from ELVIS sims.
    Subhalos are occupied via predict_satellites.py. Satellites are realized in the DES
    data via ../simSatellite.py. 
    Detection significance is evaluated using simple, but skipping the peak-finding step
    and forcing (lon_peak, lat_peak) = (ra, dec) of the satellite, and only searching at
    the single (actual) distance modulus. 

    Satellites are injected at (ra, dec) = (4.87, -38.44), to be in a similar footprint
    location to the NGC 55 satellite without overlapping it. 
    """
    (ra, dec) = (4.87, -38.44)
    region = simple.survey.Region(survey, ra, dec)

    sig_arr = np.zeros(len(sats))
    for i,sat in enumerate(sats):
        if sat['M_r'] <= -12: # Too bright, don't bother realizing:
            sig_arr[i] = 37.5
            continue

        sigs = []
        for _ in range(n_trials):
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
        sig_arr[i] = np.mean(sigs)
        percent.bar(i+1, len(sats))

    sats['sig'] = sig_arr
    if outname is not None:
        fits.writeto(outname, np.array(sats), overwrite=True)

    return sats


def count_skymap(pair, footprint, sig_table, sats, sat_cut=None, psi=None, disruption=None):
    subprocess.call('mkdir -p realizations/{}/skymaps'.format(pair).split())

    if psi is None and pair=='RJ':
        psi = np.radians(66.)
    if psi is None and pair=='TL':
        psi = np.radians(330.)
    psideg = int(round(np.degrees(psi),0))

    sat_ras, sat_decs = sats.ra_dec(psi)
    if sat_cut is not None:
        sat_ras, sat_decs = sat_ras[sat_cut], sat_decs[sat_cut]
    sigmas = sig_table['sig']

    pix = ugali.utils.healpix.angToPix(4096, sat_ras, sat_decs, nest=True)
    footprint_cut = (footprint[pix] > 0)
    detectable_cut = (sigmas >= 6.0)

    print "psi = {}: {} total sats, {} detectable".format(psideg, sum(footprint_cut), sum(footprint_cut & detectable_cut))

    plt.figure(figsize=(12,8)) 
    smap = skymap.Skymap(projection='mbtfpq', lon_0=0)
    cmap=plot_utils.shiftedColorMap('seismic_r', min(sigmas), max(sigmas), 6)
    markers = np.tile('o', len(sigmas))
    markers[footprint_cut] = '*'
    sizes = np.tile(10.0, len(sigmas))
    sizes[footprint_cut] = 50.0
    def custom_scatter(smap,x,y,markers,**kwargs):
        sc = smap.scatter(x,y,**kwargs)
        paths=[]
        for marker in markers:
            marker_obj = matplotlib.markers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
        return sc
    custom_scatter(smap, sat_ras, sat_decs, c=sigmas, cmap=cmap, latlon=True, s=sizes, markers=markers, edgecolors='k', linewidths=0.2)
    plt.colorbar()


    #Add DES polygon
    des_poly = np.genfromtxt('/Users/mcnanna/Research/y3-mw-sats/data/round19_v0.txt',names=['ra','dec'])
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
    # Disruption effets
    if disruption is not None:
        survival_cut = sats['prob'] > disruption
        if sat_cut is not None:
            survival_cut = survival_cut[sat_cut]
        print "{} disrupted sats in footprint".format(np.count_nonzero(~survival_cut & footprint_cut))
        smap.scatter(sat_ras[~survival_cut], sat_decs[~survival_cut], latlon=True, marker='x', s=30, c='k')
        plt.title('$\psi = {}^{{\circ}}$; {} total sats in footprint, {} detectable, {} disrupted'.format(psideg, sum(footprint_cut), sum(footprint_cut & detectable_cut), sum(~survival_cut & footprint_cut)))
    else:
        plt.title('$\psi = {}^{{\circ}}$; {} total sats in footprint, {} detectable'.format(psideg, sum(footprint_cut), sum(footprint_cut & detectable_cut)))
    plt.savefig('realizations/{0}/skymaps/{0}_skymap_psi={1:0>3d}.png'.format(pair, psideg), bbox_inches='tight')
    plt.close()

    return (footprint_cut, detectable_cut, survival_cut)


def rotated_skymaps(pair, footprint, sig_table, sats, sat_cut=None, n_rotations=60, disruption=None):
    subprocess.call('mkdir -p realizations/{}/skymaps'.format(pair).split())

    cut_results = []
    for i in range(n_rotations):
        psi = 2*np.pi * float(i)/n_rotations
        cuts = count_skymap(pair, footprint, sig_table, sats, sat_cut=sat_cut, psi=psi, disruption=disruption)
        cut_results.append(cuts)

    # Merge skymaps into a .gif
    print '\nCreating .gif...'
    subprocess.call("convert -delay 30 -loop 0 realizations/{0}/skymaps/*.png realizations/{0}/{0}_skymap.gif".format(pair).split())
    print 'Done!'

    # Save results
    cut_results = np.array(cut_results)
    np.save('realizations/{0}/{0}_skymap_cuts'.format(pair),cut_results)


def summary_plots(pair, sats=None, psi=None):
    skymap_cuts = np.load('realizations/{0}/{0}_skymap_cuts.npy'.format(pair))

    if args['pair'] == 'RJ':
        title = 'Romeo \& Juliet'
    elif args['pair'] == 'TL':
        title = 'Thelma \& Louise'
    elif args['pair'] == 'RR':
        title = 'Romulus \& Remus'

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

    #total_sats = [sum(cuts[0]) for cuts in skymap_cuts]
    total_sats = [sum(footprint_cut) for footprint_cut, detectable_cut in skymap_cuts]
    hist(total_sats, 'Satellites in footprint', 'total_sats_hist')
    #detectable_sats = [sum(cuts[0]&cuts[1]) for cuts in skymap_cuts]
    detectable_sats = [sum(footprint_cut & detectable_cut) for footprint_cut, detectable_cut in skymap_cuts]
    hist(detectable_sats, 'Detectable satellites in foorprint', 'detectable_sats_hist')

    if sats is not None:
        # Can also do distances
        if psi is not None:
            try:
                idx = range(0,360,6).index(psi)
            except ValueError:
                psi_new = int(round(psi/6.)*6)
                print("psi={} not a multiple of 6, rounding to psi={}".format(psi, psi_new))
                psi = psi_new
                idx = range(0,360,6).index(psi)

            footprint_cut = skymap_cuts[idx][0]
            detectable_cut = skymap_cuts[idx][1]

            bins = np.arange(300, 2030, 75)
            plt.hist(sats['distance'][footprint_cut], bins=bins, label="In footprint")
            plt.hist(sats['distance'][footprint_cut & detectable_cut], bins=bins, label="Detectable in footprint")
            plt.title(title)
            plt.savefig('realizations/{}/distance_hist_psi={}.png'.format(pair, psi))
            plt.close()

        else:
            # Sum over different psis, plot pdf
            footprint_distances = []
            and_detectable_distances = []
            for footprint_cut, detectable_cut in skymap_cuts:
                footprint_distances.append(sats['distance'][footprint_cut])
                and_detectable_distances.append(sats['distance'][footprint_cut & detectable_cut])
            
            bins = np.arange(300,2030,75)
            plt.hist(np.concatenate(footprint_distances), bins=bins, label="In footprint", density=True, alpha=0.6)
            plt.hist(np.concatenate(and_detectable_distances), bins=bins, label="Detectable in footprint", density=True, alpha=0.6)
            plt.title(title)
            plt.legend()
            plt.savefig('realizations/{}/distance_hist.png'.format(pair))



if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pair', default='RJ')
    p.add_argument('-f', '--fname', default='sats_table_{}.fits')
    p.add_argument('-t', '--table', action='store_true')
    p.add_argument('--n_trials', default=1, type=int)
    p.add_argument('--config')
    p.add_argument('--psi', default=None, type=float)
    p.add_argument('--prob', default=None, type=float, help="Survival probabiliy threshold")
    p.add_argument('-r', '--rotations', action='store_true')
    p.add_argument('-s', '--summary', action='store_true')
    p.add_argument('-m', '--main', action='store_true')
    args = vars(p.parse_args())
    if '{}' in args['fname']:
        args['fname'] = args['fname'].format(args['pair'])
    if '.fits' not in args['fname']:
        args['fname'] += '.fits'

    # Cut satellites to those outside MW virial radius, but < 2 Mpc distance. 
    if args['table'] or args['psi'] or args['rotations'] or args['summary']:
        sats = predict_satellites.Satellites(args['pair'])
        close_cut = (sats.distance > 300)
        far_cut = (sats.distance < 2000)
        cut = close_cut & far_cut
    
    if args['table']:
        if args['config'] is None:
            raise ValueError("Config required for significance table creation")
        with open(args['config'], 'r') as ymlfile:
            cfg = yaml.load(ymlfile)
            survey = simple.survey.Survey(cfg)
            inputs = load_data.Inputs(cfg)

        subprocess.call('mkdir -p realizations/{}'.format(args['pair']).split())
        print '\n Excluding {} satelites closer than 300 kpc and {} beyond 2000 kpc'.format(sum(~close_cut), sum(~far_cut))
        print "\nCalculating significances for {} satellites...".format(sum(cut))
        create_sigma_table(sats[cut], inputs, survey, outname='realizations/{}/{}'.format(args['pair'],args['fname']), n_trials=args['n_trials'])


    if (args['psi'] is not None) or args['rotations']:
        sigma_table = fits.open('realizations/{}/{}'.format(args['pair'], args['fname']))[1].data
        subprocess.call('mkdir -p realizations/{}/skymaps'.format(args['pair']).split())
        footprint = ugali.utils.healpix.read_map('~/Research/y6/v2_y6_dwarf_search/candidate-analysis/data/y6_gold_v2_footprint.fits', nest=True)
    if args['psi'] is not None:
        count_skymap(args['pair'], footprint, sigma_table, sats, sat_cut=cut, psi=np.radians(args['psi']), disruption=args['prob'])
    if args['rotations']:
        rotated_skymaps(args['pair'], footprint, sigma_table, sats, sat_cut=cut, n_rotations=60, disruption=args['prob'])

    if args['summary']:
        if args['psi'] is not None:
            summary_plots(args['pair'], sats[cut], args['psi'])
        else:
            summary_plots(args['pair'], sats[cut])


def main(): # Needs work on arguments
    sats = predict_satellites.Satellites(pair)
    if pair == 'RJ':
        psi = np.radians(66)
    elif pair == 'TL':
        psi = np.radians(330)
    psideg = int(round(np.degrees(psi),0))
    sat_ras, sat_decs = sats.ra_dec(psi)

    footprint = ugali.utils.healpix.read_map('~/Research/y6/v2_y6_dwarf_search/candidate-analysis/data/y6_gold_v2_footprint.fits', nest=True)
    pix = ugali.utils.healpix.angToPix(4096, sat_ras, sat_decs, nest=True)
    footprint_cut = footprint[pix] > 0

    outname = ''
    table = create_sigma_table(sats[cut], inputs, survey, outname=outname, n_trials=1)
    # Add ra and dec to the table
    table['ra'] = sat_ras[cut & footprint_cut]
    table['dec'] = sat_decs[cut & footprint_cut]

    fits.writeto('realizations/{0}/sats_table_{0}_psi={1}.fits'.format(pair, psideg), np.array(table), overwrite=True)

