import argparse
import numpy as np
import yaml
import astropy.io.fits as fits
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle

import simple.survey
import simple.search
import ugali.utils.projector

import load_data
import simSatellite
import utils

import percent
import plot_utils

matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['xtick.labelsize'] = 14
matplotlib.rcParams['ytick.labelsize'] = 14
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
plt.ion()


def create_sigma_matrix(args, inputs, region, abs_mags, r_physicals, distance, outname=None, n_trials=1):
    # r_physicals in parsecs
    n_m = len(abs_mags)
    n_r = len(r_physicals)

    sigma_fits = []
    counter = 0
    for i in range(n_m):
        for j in range(n_r):
            m, r = abs_mags[i], r_physicals[j]

            if m < -11.99: # Don't bother simulating, it's so bright that sig = max
                "Skipping due to abs_mag <= -12"
                sigma = 37.5
                sigma_fits.append((m, r, sigma))
                counter += 1
                percent.bar(counter, n_m*n_r)
                continue
            
            sigmas = []
            for k in range(n_trials):
                sim = simSatellite.SimSatellite(inputs, args['ra'], args['dec'], distance, m, r)
                data, n_stars = simSatellite.inject(region, sim)
                region.data = data
                sig = simSatellite.search(region, data, mod=ugali.utils.projector.distanceToDistanceModulus(distance))
                sigmas.append(sig)
                if k == 0 and sig > 37.4 : # No need to keep simulating if it's just going to max out
                    print "Stopping trials due to high significance"
                    break

            sigma = np.mean(sigmas)
        
            sigma_fits.append((m, r, sigma))

            counter += 1
            percent.bar(counter, n_m*n_r)


    dtype = [('abs_mag',float), ('r_physical',float), ('sigma',float)]
    sigma_fits = np.array(sigma_fits, dtype=dtype)
    if outname is not None:
        fits.writeto(outname+'.fits', sigma_fits, overwrite=True)

    return sigma_fits


def plot_sigma_matrix(fname, distance):
    # Mainly copied from ~/Research/y6/far_out/significance.py. An even more general version is found there

    dic = {'distance': {'label':'$D$', 'conversion': lambda d:int(d), 'unit':'kpc', 'scale':'linear', 'reverse':False}, 
           'abs_mag': {'label':'$M_V$', 'conversion': lambda v:round(v,1), 'unit':'mag', 'scale':'linear', 'reverse':True},
           'r_physical': {'label':'$r_{1/2}$', 'conversion': lambda r:int(round(r,0)), 'unit':'pc', 'scale':'log', 'reverse':False},
           'stellar_mass': {'label':'$M_*$', 'conversion': lambda m: '$10^{{{}}}$'.format(round(np.log10(m),1)), 'unit':'$M_{\odot}$', 'scale':'log', 'reverse':False},
           }
    def is_near(arr, val, e=0.001):
        return np.array([val-e < a < val+e for a in arr])

    if '.fits' not in fname:
        fname += '.fits'
    table = fits.open(fname)[1].data

    x, y = 'r_physical', 'abs_mag'
    x_vals = sorted(set(table[x]), reverse=dic[x]['reverse'])
    y_vals = sorted(set(table[y]), reverse=dic[y]['reverse'])
    mat_sigma = np.zeros((len(x_vals), len(y_vals)))
    for i, x_val in enumerate(x_vals):
        for j, y_val in enumerate(y_vals):
            line = table[is_near(table[x], x_val) & is_near(table[y], y_val)]
            mat_sigma[i,j] = line['sigma']
    

    plt.figure(figsize=((len(x_vals)+3)*0.8, (len(y_vals)/2.0)*0.6))
    
    mn, mid, mx = 0, 6, 37.5
    norm = matplotlib.colors.Normalize
    cmap = plot_utils.shiftedColorMap('bwr_r', mn, mx, mid)

    plt.pcolormesh(mat_sigma.T, cmap=cmap, norm=norm(vmin=mn, vmax=mx))
    ax = plt.gca()

    xticks = np.arange(len(x_vals)) + 0.5
    ax.set_xticks(xticks)
    ax.set_xticklabels(map(dic[x]['conversion'], x_vals))
    plt.xlabel('{} ({})'.format(dic[x]['label'], dic[x]['unit']))

    yticks = np.arange(len(y_vals)) + 0.5
    yticks = yticks[1::2] # Every other tick to avoid crowding the axis
    ax.set_yticks(yticks)
    #ax.set_yticks(yticks[1::2]) # Every other tick 
    yticklabels = map(dic[y]['conversion'], y_vals)
    yticklabels = yticklabels[1::2] # Every other label
    ax.set_yticklabels(yticklabels)
    #ax.set_yticklabels(yticklabels[1::2]) # Every other label
    plt.ylabel('{} ({})'.format(dic[y]['label'], dic[y]['unit']))

    fixed = 'distance'
    title = "{} = {} {}".format(dic['distance']['label'], dic['distance']['conversion'](distance), dic['distance']['unit'])
    
    cbar_label = '$\sigma$'

    abs_mags = np.array( sorted(set(table['abs_mag']), reverse=True) )
    stellar_masses = utils.mag_to_mass(abs_mags)
    if x == 'abs_mag':
        twin_ax = ax.twiny()
        twin_ax.set_xticks(list(xticks) + [xticks[-1]+0.5]) # Have to add an extra on the end to make it scale right
        twin_ax.set_xticklabels(map(dic['stellar_mass']['conversion'], stellar_masses) + [''])
        twin_ax.set_xlabel('{} ({})'.format(dic['stellar_mass']['label'], dic['stellar_mass']['unit']))
        plt.subplots_adjust(top=0.85) # Make more vertical room 
        plt.colorbar(label=cbar_label)
        plt.title(title, y=1.12) # Push title above upper axis labels
    elif y == 'abs_mag':
        twin_ax = ax.twinx()
        twin_ax.set_yticks(list(yticks) + [yticks[-1]+0.5])
        yticklabels = map(dic['stellar_mass']['conversion'], stellar_masses) + ['']
        twin_ax.set_yticklabels(yticklabels[1::2])
        #twin_ax.set_yticklabels(map(dic['stellar_mass']['conversion'], stellar_masses) + [''])
        twin_ax.set_ylabel('{} ({})'.format(dic['stellar_mass']['label'], dic['stellar_mass']['unit']))
        plt.colorbar(label=cbar_label, pad=0.10) # Move colorbar right to make room for axis labels
        plt.title(title)

    xmin, xmax = (x_vals[0], x_vals[-1]) if not dic[x]['reverse'] else (x_vals[-1], x_vals[0])
    ymin, ymax = (y_vals[0], y_vals[-1]) if not dic[y]['reverse'] else (y_vals[-1], y_vals[0])  

    def transform(value, axis_vals, log=False):
        if log:
            axis_vals = np.log10(axis_vals)
            value = np.log10(value)
        delta = axis_vals[1] - axis_vals[0]
        mn = axis_vals[0] - delta/2.
        mx = axis_vals[-1] + delta/2.
        return ((value-mn)/(mx-mn)) * len(axis_vals)


    
    # Place known sats on plot (from mw_sats_master.csv):
    plt.sca(ax)
    translation = {'distance':'distance_kpc', 'abs_mag':'m_v', 'r_physical':'a_physical'}
    dwarfs = load_data.Satellites().dwarfs
    cut = xmin < dwarfs[translation[x]]
    cut &= dwarfs[translation[x]] < xmax
    cut &= ymin < dwarfs[translation[y]]
    cut &= dwarfs[translation[y]] < ymax
    #cut &= np.array(['des' in survey for survey in dwarfs['survey']])

    ngc_sat_names = ['ESO410-005', 'ESO294-010', 'UGCA438']
    ngc_cut = np.array([name in ngc_sat_names for name in dwarfs['name']])
    ngc_sats = dwarfs[cut & ngc_cut]

    dwarfs = dwarfs[cut & ~ngc_cut]

    sat_xs = transform(dwarfs[translation[x]], x_vals, dic[x]['scale']=='log')
    sat_ys = transform(dwarfs[translation[y]], y_vals, dic[y]['scale']=='log')
    plt.scatter(sat_xs, sat_ys, color='k')
    """
    #down = ['Boo II', 'Pic I', 'Ret II', 'Phe II', 'Leo V', 'Hyi I', 'UMa II']
    #left = ['Phe II', 'Gru II', 'Psc II', 'Com', 'Sex', 'UMa II', 'Pic II']

    down = ['Sex', 'UMi', 'Phe II', 'Hor II', 'Boo II', 'Pic II', 'Peg III', 'Tuc IV', 'Gru II', 'Psc II', 'Com', 'Col I', 'Hyd II']
    left = ['Dra', 'UMi', 'Pic I', 'Phe II', 'Ret II', 'Peg III', 'Leo V', 'Hyi I', 'Psc II', 'Hyd II', 'Sgr II', 'CVn II', 'Leo IV']

    for i, d in enumerate(dwarfs):
        xy = (sat_xs[i], sat_ys[i])
        xytext = [3,3]
        ha = 'left'
        va = 'bottom'
        if d['abbreviation'] in down:
            va = 'top'
            xytext[1] = -3
        if d['abbreviation'] in left:
            ha = 'right'
            xytext[0] = -3
        plt.annotate(d['abbreviation'], xy, textcoords='offset points', xytext=xytext, ha=ha, va=va, fontsize=10,
                     bbox = dict(facecolor='white', boxstyle='round,pad=0.2'))
    """
    names = ['Antlia II', 'Crater II', 'Sagittarius']
    named_cut = [d['name'] in names for d in dwarfs]
    down = ['Crater II']
    for i, d in enumerate(dwarfs[named_cut]):
        xy = (sat_xs[named_cut][i], sat_ys[named_cut][i])
        xytext = [3,3]
        ha = 'left'
        va = 'bottom'
        if d['name'] in down:
            va = 'top'
            xytext[1] = -3

        plt.annotate(d['name'], xy, textcoords='offset points', xytext=xytext, ha=ha, va=va, fontsize=10, bbox=dict(facecolor='white', boxstyle='round,pad=0.2'))

    sat_xs = transform(ngc_sats[translation[x]], x_vals, dic[x]['scale']=='log')
    sat_ys = transform(ngc_sats[translation[y]], y_vals, dic[y]['scale']=='log')
    #plt.scatter(sat_xs, sat_ys, color='k', marker='*', s=100)
    plt.scatter(sat_xs, sat_ys, color='k', marker='x')
    down = ['UGCA438']
    left = []
    for i, d in enumerate(ngc_sats):
        xy = (sat_xs[i], sat_ys[i])
        xytext = [3,3]
        ha = 'left'
        va = 'bottom'
        if d['name'] in down:
            va = 'top'
            xytext[1] = -3
        if d['name'] in left:
            ha = 'right'
            xytext[0] = -3

        if d['name'] == 'ESO294-010':
            xytext = [17, -15]
            plt.annotate(d['name'], xy, textcoords='offset points', xytext=xytext, ha=ha, va='top', fontsize=10, bbox=dict(facecolor='white', boxstyle='round,pad=0.2'),
                    arrowprops=dict(arrowstyle='-'))
        elif d['name'] == 'ESO410-005':
            xytext = [30, 12]
            plt.annotate(d['name'], xy, textcoords='offset points', xytext=xytext, ha=ha, va=va, fontsize=10, bbox=dict(facecolor='white', boxstyle='round,pad=0.2'),
                    arrowprops=dict(arrowstyle='-'))

        else: 
            plt.annotate(d['name'], xy, textcoords='offset points', xytext=xytext, ha=ha, va=va, fontsize=10, bbox=dict(facecolor='white', boxstyle='round,pad=0.2'))



    # Place known LG dwarfs on plot (from McConnachie2015):
    plt.sca(ax)
    #TODO: r_physical vs a_physical: a_physical seems to be more commonly used, and make the figure look better imo
    translation = {'distance':'distanc', 'abs_mag':'m_v', 'r_physical':'a_physical'}
    dwarfs = load_data.McConnachie15().data
    cut = xmin < dwarfs[translation[x]]
    cut &= dwarfs[translation[x]] < xmax
    cut &= ymin < dwarfs[translation[y]]
    cut &= dwarfs[translation[y]] < ymax
    #cut &= np.array(['des' in survey for survey in dwarfs['survey']])
    dwarfs = dwarfs[cut]

    non_sats = ['ESO 294 -G 010', 'Cetus', 'KKR 25', 'DDO 113', 'Aquarius', 'Antlia', 'KKH 86', 'LGS 3', 'Phoenix', 'Antlia B', 'Tucana', 'KKR 3', 'Leo T']

    non_sats = ['LGS 3','Phoenix','Cetus','Pegasus dIrr','Leo T','Leo A','Aquarius','Tucana','Sagittarius dIrr','UGC 4879','Antlia B','Antlia','KKR 25','KKH 98','UKS 2323-326','KKR 3','KKs3','GR 8','UGC 9128','UGC 8508','IC 3104','UGCA 86','DDO 99','KKH 86','DDO 113']

    non_sat_cut = np.array([d['name'] in non_sats for d in dwarfs])
    #and_cut = np.array(['And' in d['name'] for d in dwarfs])
    and_cut = np.array([('And' in d['name']) for d in dwarfs])

    sat_xs = transform(dwarfs[translation[x]], x_vals, dic[x]['scale']=='log')
    sat_ys = transform(dwarfs[translation[y]], y_vals, dic[y]['scale']=='log')
    plt.scatter(sat_xs[non_sat_cut], sat_ys[non_sat_cut], color='k', marker='x')
    plt.scatter(sat_xs[and_cut], sat_ys[and_cut], color='k', marker='s', zorder=102)

    """
    for i, d in enumerate(dwarfs):
        xy = (sat_xs[i], sat_ys[i])
        xytext = [3,3]
        ha = 'left'
        va = 'bottom'
        #if d['abbreviation'] in down:
        #    va = 'top'
        #    xytext[1] = -3
        #if d['abbreviation'] in left:
        #    ha = 'right'
        #    xytext[0] = -3
        plt.annotate(d['name'][-7:], xy, textcoords='offset points', xytext=xytext, ha=ha, va=va, fontsize=10,
                     bbox = dict(facecolor='white', boxstyle='round,pad=0.2'))
    """
    names = ['Andromeda XIX', 'Andromeda XXIII', 'And XXXII', 'Andromeda XXI', 'Andromeda II'] #, 'Andromeda XXV']
    named_cut = [d['name'] in names for d in dwarfs]
    down = ['Andromeda XXI']
    for i, d in enumerate(dwarfs[named_cut]):
        xy = (sat_xs[named_cut][i], sat_ys[named_cut][i])
        xytext = [3,3]
        ha = 'left'
        va = 'bottom'
        if d['name'] in down:
            va = 'top'
            xytext[1] = -3


        if 'And' in d['name']:
            annote = 'And ' + d['name'].split(' ')[-1] # And + Roman Numeral
        else:
            annote = d['name']

        if d['name'] == 'Andromeda XIX':
            xytext = [6, 20]
            plt.annotate(annote, xy, textcoords='offset points', xytext=xytext, ha='center', va=va, fontsize=10, bbox=dict(facecolor='white', boxstyle='round,pad=0.2'), zorder=104,
                    arrowprops=dict(arrowstyle='-'))
        else:
            plt.annotate(annote, xy, textcoords='offset points', xytext=xytext, ha=ha, va=va, fontsize=10, bbox=dict(facecolor='white', boxstyle='round,pad=0.2'), zorder=104)
   
    # Add approximate candidate location
    #cand_x, cand_y = 3300, -10
    cand_x, cand_y = 3256, -7.9 # Based on values from mcmc_tight_qual
    cand_x = transform(cand_x, x_vals, dic[x]['scale']=='log')
    cand_y = transform(cand_y, y_vals, dic[y]['scale']=='log')
    plt.scatter(cand_x, cand_y, marker='*', color='yellow', edgecolor='k', s=700, zorder=101)
    #plt.annotate('Approximate candidate location', (cand_x, cand_y), textcoords='offset points', xytext=[-9,4], ha='right', va='bottom', fontsize=14, zorder=100, bbox = dict(facecolor='white', boxstyle='round,pad=0.2'))
    

    outname = '{}_vs_{}__'.format(x, y) + '{}={}'.format('distance', int(distance))
    plt.savefig('{}.eps'.format(outname), bbox_inches='tight')
    plt.savefig('{}.png'.format(outname), bbox_inches='tight')
    plt.close()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--ra', required=True, type=float)
    parser.add_argument('--dec', required=True, type=float)
    parser.add_argument('--distance', type=float, default=2000, help="kpc")
    #parser.add_argument('--distance', required=True, type=float)
    #parser.add_argument('--mod', required=True, type=float)
    #parser.add_argument('--r', required=True, type=float, help="Aperture size (arcmin)")
    #parser.add_argument('--n_obs', required=True, type=float)
    #parser.add_argument('--id', required=False, type=int, default=0)
    #parser.add_argument('--plots', action='store_true')
    args = vars(parser.parse_args())


    with open(args['config'], 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
        survey = simple.survey.Survey(cfg)

    inputs = load_data.Inputs(cfg)
    region = simple.survey.Region(survey, args['ra'], args['dec'])


    abs_mags = np.arange(-2.5, -14.5, -0.5)
    log_r_physical_pcs = np.arange(1.4, 3.8, 0.2)
    r_physicals = 10**log_r_physical_pcs

    outname = 'sigma_table_10trials__{}kpc'.format(int(args['distance']))
    create_sigma_matrix(args, inputs, region, abs_mags, r_physicals, args['distance'], outname=outname, n_trials=10)

    plot_sigma_matrix(outname, args['distance'])


