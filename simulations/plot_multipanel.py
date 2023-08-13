import argparse
import numpy as np
import scipy.interpolate
import yaml
import astropy.io.fits as fits
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import simple.survey
import simple.search
import ugali.utils.projector

import load_data
import simSatellite
import utils

import percent
import plot_utils

matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
plt.ioff()


def plot_sensitivity(fname, distances, nrows, ncols, typ='prob', density_table=None):
    dic = {'distance': {'label':'$D$', 'conversion': lambda d:int(d), 'unit':'kpc', 'scale':'linear', 'reverse':False}, 
           'abs_mag': {'label':'$M_V$', 'conversion': lambda v:round(v,1), 'unit':'mag', 'scale':'linear', 'reverse':True},
           #'a_physical': {'label':'$a_{1/2}$', 'conversion': lambda r:int(round(r,0)), 'unit':'pc', 'scale':'log', 'reverse':False},
           'a_physical': {'label':'$\log_{10}a_{1/2}$', 'conversion': lambda r:round(np.log10(r),1), 'unit':'pc', 'scale':'log', 'reverse':False},
           'stellar_mass': {'label':'$M_*$', 'conversion': lambda m: '$10^{{{}}}$'.format(round(np.log10(m),1)), 'unit':'$M_{\odot}$', 'scale':'log', 'reverse':False},
           }
    def is_near(arr, val, e=0.001):
        return np.array([val-e < a < val+e for a in arr])

    full_table = fits.open(fname)[1].data
    if density_table is not None:
        full_densities = fits.open(density_table)[1].data

    fig, axs = plt.subplots(nrows, ncols, figsize=(17, 10)) # Could take figsize formula from plot_sigma_matrix and mutiply by nrows and nccols
    plt.subplots_adjust(wspace=0.38)
    axes = axs.flatten()
    for idx in range(len(distances)):
        distance = distances[idx]
        ax = axes[idx]

        distance_cut = is_near(full_table['distance'], distance)
        table = full_table[distance_cut]
        if density_table is not None:
            density_distance_cut = is_near(full_densities['distance'], distance)
            densities = full_densities[density_distance_cut]

        x, y = 'a_physical', 'abs_mag'
        x_vals = sorted(set(table[x]), reverse=dic[x]['reverse'])
        y_vals = sorted(set(table[y]), reverse=dic[y]['reverse'])
        mat_result = np.zeros((len(x_vals), len(y_vals)))
        for i, x_val in enumerate(x_vals):
            #if density_table is not None:
            #    # Create interpolation function to estimate abs_mags between a_physical values
            #    a_cut = is_near(table[x], x_val)
            #    prob_interp = scipy.interpolate.interp1d(table[a_cut][y], table[a_cut][typ])
            for j, y_val in enumerate(y_vals):
                line = table[is_near(table[x], x_val) & is_near(table[y], y_val)]
                try:
                    mat_result[i, j] = line[typ]
                except ValueError:
                    print(distance, y_val, x_val)
                    mat_result[i, j] = 0

                #if density_table is not None:
                #    # Override above mat_result insertion by adjusting for blended abs_mag 
                #    density = densities[is_near(densities[x], x_val) & is_near(densities[y], y_val)]['density'][0]
                #    mag_reduction = max(0, 0.39*np.log(density) + 2.21) # fit is from blending.ipynb based on Jonah's tests
                #    try:
                #        mat_result[i, j] = prob_interp(y_val + mag_reduction)
                #    except ValueError:
                #        #print(x_val, y_val, round(y_val + mag_reduction,1))
                #        mat_result[i, j] = 0

        # Convert from log10(a_physical), which is what's stored in the tables, to just a_physical
        x_vals = 10**np.array(x_vals) 
    
        plt.sca(ax)
        if typ == 'sigma':
            # Fudging
            floor = np.min(mat_result)
            new_floor = 2.2
            for i in range(len(mat_result)):
                for j in range(len(mat_result[0])):
                    if mat_result[i,j] < 6:
                        mat_result[i,j] = (mat_result[i,j]-floor)/(6-floor)*(6-new_floor) + new_floor

            mn, mid, mx = 0, 6, 37.5
            norm = matplotlib.colors.Normalize
            cmap = plot_utils.shiftedColorMap('bwr_r', mn, mx, mid)
            im = plt.pcolormesh(mat_result.T, cmap=cmap, norm=norm(vmin=mn, vmax=mx))
            cbar_label = '$\sigma$'

        elif typ == 'prob':
            cmap = plot_utils.truncatedColorMap('Blues_r', n_min=50)
            im = plt.pcolormesh(mat_result.T, cmap=cmap)
            cbar_label = 'Detection Probability'

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

        #Draw line at M_V = -12
        if x == 'abs_mag':
            pass
        if y == 'abs_mag':
            yval = transform(-12.0, y_vals, dic[y]['scale']=='log')
            #ax.axhline(yval, linestyle='--', color='k')
            width = transform(xmax, x_vals, dic[x]['scale']=='log')+0.5

            if dic[y]['reverse'] == True:
                bottom = yval + 0.5 # Extra 0.5 because the -12 square ends above the -12 tick
                top = transform(ymin, y_vals, dic[y]['scale']=='log') + 0.5
                height = top - bottom
            matplotlib.rcParams["hatch.color"] = 'k'
            rect = Rectangle((0, bottom), width, height, hatch='/', facecolor = 'none', zorder=50)
            #ax.add_patch(rect)

        # Add distance label
        ax.text(0.4, 22, "$D = {:>2}$ Mpc".format(distance/1000.), bbox=dict(facecolor='white', boxstyle='round,pad=0.2'), zorder=104)

        ### Blending, visualization option 2
        if density_table is not None:
            full_densities = fits.open(density_table)[1].data
            density_distance_cut = is_near(full_densities['distance'], distance)
            densities = full_densities[density_distance_cut]

            blended_x = []
            blended_y = []

            for x_val in x_vals: # a_physicals
                xval = dic[x]['conversion'](x_val) # Converts to log(a_physical) which is what's stored in the tables
                a_cut = is_near(table[x], xval)
                prob_interp = scipy.interpolate.interp1d(table[a_cut][y], table[a_cut][typ])
                for y_val in y_vals: # abs_mags
                    density = densities[is_near(densities[x], xval) & is_near(densities[y], y_val)]['density'][0]
                    mag_reduction = max(0, 0.39*np.log(density) + 2.21) # fit is from blending.ipynb based on Jonah's tests
                    try:
                        recovery_prob = prob_interp(y_val + mag_reduction)
                    except ValueError:
                        recovery_prob = 0
                    noblend_prob = table[is_near(table[x], xval) & is_near(table[y], y_val)][typ]

                    if (noblend_prob > 0.5) and (recovery_prob < 0.5):
                        blended_x.append(x_val)
                        blended_y.append(y_val)

            blended_x = transform(blended_x, x_vals, dic[x]['scale'] == 'log')
            blended_y = transform(blended_y, y_vals, dic[y]['scale'] == 'log')
            matplotlib.rcParams["hatch.color"] = '0.1'
            for x_center, y_center in zip(blended_x, blended_y):
                rect = Rectangle((x_center-0.5, y_center-0.5), 1, 1, facecolor='none', hatch='X', zorder=51)
                ax.add_patch(rect)


        #TODO maybe: LSST curve

        ### Place known sats on plot (from mw_sats_master.csv):
        #plt.sca(ax)
        translation = {'distance':'distance_kpc', 'abs_mag':'m_v', 'a_physical':'a_physical'}
        dwarfs = load_data.Satellites().dwarfs
        cut = xmin < dwarfs[translation[x]]
        cut &= dwarfs[translation[x]] < xmax
        cut &= ymin < dwarfs[translation[y]]
        cut &= dwarfs[translation[y]] < ymax
        # Add distance cut
        #print(max(dwarfs[cut][translation['distance']]))
        #cut &= (dwarfs[translation['distance']] < distance) & (dwarfs[translation['distance']] > distance-500)
        if distance == 500:
            cut &= dwarfs[translation['distance']] < 750
        elif distance == 2000:
            cut &= dwarfs[translation['distance']] > 1750
        else:
            cut &= (dwarfs[translation['distance']] > distance-250)  & (dwarfs[translation['distance']] < distance+250)
        #cut &= np.array(['des' in survey for survey in dwarfs['survey']])
        dwarfs = dwarfs[cut]

        sat_xs = transform(dwarfs[translation[x]], x_vals, dic[x]['scale']=='log')
        sat_ys = transform(dwarfs[translation[y]], y_vals, dic[y]['scale']=='log')
        plt.scatter(sat_xs, sat_ys, s=20, color='k', zorder=100)

        # Label the big ones
        names = ['Antlia II', 'Crater II']#, 'Sagittarius']
        named_cut = [d['name'] in names for d in dwarfs]
        down = ['Crater II']
        left = ['Antlia II']
        for i, d in enumerate(dwarfs[named_cut]):
            xy = (sat_xs[named_cut][i], sat_ys[named_cut][i])
            xytext = [3,3]
            ha = 'left'
            va = 'bottom'
            if d['name'] in down:
                va = 'top'
                xytext[1] = -3
            if d['name'] in left:
                ha = 'right'
                xytext[0] = -3

            if 'Crater' in d['name']: annote = 'Crt ' + d['name'].split(' ')[-1] # abbv + Roman Numeral
            elif 'Antlia' in d['name']:  annote = 'Ant ' + d['name'].split(' ')[-1] # abbv + Roman Numeral
            else: annote = d['name']
            plt.annotate(annote, xy, textcoords='offset points', xytext=xytext, ha=ha, va=va, fontsize=9, bbox=dict(facecolor='white', boxstyle='round,pad=0.2'), zorder=104)


        # Place known LG dwarfs on plot (from McConnachie2015):
        #plt.sca(ax)
        #TODO: r_physical vs a_physical: a_physical seems to be more commonly used, and make the figure look better imo
        translation = {'distance':'distance', 'abs_mag':'m_v', 'a_physical':'a_physical'}
        dwarfs = load_data.McConnachie15().data
        cut = xmin < dwarfs[translation[x]]
        cut &= dwarfs[translation[x]] < xmax
        cut &= ymin < dwarfs[translation[y]]
        cut &= dwarfs[translation[y]] < ymax
        #cut &= np.array(['des' in survey for survey in dwarfs['survey']])
        # Add distance cut
        #print(max(dwarfs[cut][translation['distance']]))
        #cut &= (dwarfs[translation['distance']] < distance) & (dwarfs[translation['distance']] > distance-500)
        if distance == 500:
            cut &= dwarfs[translation['distance']] < 750
        elif distance == 2000:
            cut &= dwarfs[translation['distance']] > 1750
        else:
            cut &= (dwarfs[translation['distance']] > distance-250)  & (dwarfs[translation['distance']] < distance+250)
        dwarfs = dwarfs[cut]

        non_sats = ['LGS 3','Phoenix','Cetus','Pegasus dIrr','Leo T','Leo A','Aquarius','Tucana','Sagittarius dIrr','UGC 4879','Antlia B','Antlia','KKR 25','KKH 98','KKR 3','KKs3','GR 8','UGC 9128','UGC 8508','IC 3104','UGCA 86','DDO 99','KKH 86','DDO 113', 'ESO 410- G 005', 'ESO 294- G 010', 'UKS 2323-326'] # UKS 2323-326 is another name for UGCA438

        non_sat_cut = np.array([d['name'] in non_sats for d in dwarfs])
        and_cut = np.array([('And' in d['name']) for d in dwarfs])
        ands = dwarfs[and_cut]

        sat_xs = transform(dwarfs[translation[x]], x_vals, dic[x]['scale']=='log')
        sat_ys = transform(dwarfs[translation[y]], y_vals, dic[y]['scale']=='log')
        plt.scatter(sat_xs[non_sat_cut], sat_ys[non_sat_cut], color='k', s=20, marker='x', zorder=101)
        plt.scatter(sat_xs[and_cut], sat_ys[and_cut], color='k', s=20, marker='s', zorder=102)

        # Label NGC55 "sats"
        #names = ['ESO 410- G 005', 'ESO 294- G 010', 'UKS 2323-326']
        #annotes = ['ESO410-005G', 'ESO294-010', 'UGCA438']
        names = ['ESO 294- G 010']
        annotes = ['ESO294-010']
        named_cut = [d['name'] in names for d in dwarfs]
        down = ['ESO 294- G 010', 'UKS 2323-326']
        left = ['ESO 294- G 010', 'UKS 2323-326']
        for i, d in enumerate(dwarfs[named_cut]):
            xy = (sat_xs[named_cut][i], sat_ys[named_cut][i])
            xytext = [3,3]
            ha = 'left'
            va = 'bottom'
            
            if d['name'] in down:
                va = 'top'
                xytext[1] = -3
            if d['name'] in left:
                ha = 'right'
                xytext[0] = -3

            if d['name'] == 'ESO 410- G 005':
                xytext = [-18, 3]
                plt.annotate(annotes[i], xy, textcoords='offset points', xytext=xytext, ha='right', va=va, fontsize=10, bbox=dict(facecolor='white', boxstyle='round,pad=0.2'), zorder=104, arrowprops=dict(arrowstyle='-'))
            else:
                pass
                plt.annotate(annotes[i], xy, textcoords='offset points', xytext=xytext, ha=ha, va=va, fontsize=10, bbox=dict(facecolor='white', boxstyle='round,pad=0.2'), zorder=104)

        # Label big m31 sats
        names = ['Andromeda XIX']#, 'Andromeda XXIII', 'And XXXII', 'Andromeda XXI', 'Andromeda II'] #, 'Andromeda XXV']
        named_cut = [d['name'] in names for d in dwarfs]
        down = ['Andromeda XXI']
        left = ['Andromeda XIX']
        for i, d in enumerate(dwarfs[named_cut]):
            xy = (sat_xs[named_cut][i], sat_ys[named_cut][i])
            xytext = [3,3]
            ha = 'left'
            va = 'bottom'
            if d['name'] in down:
                va = 'top'
                xytext[1] = -3
            if d['name'] in left:
                ha = 'right'
                xytext[0] = -3

            if 'And' in d['name']:
                annote = 'And ' + d['name'].split(' ')[-1] # And + Roman Numeral
            else:
                annote = d['name']

            #if d['name'] == 'Andromeda XIX':
            #    xytext = [6, 20]
            #    plt.annotate(annote, xy, textcoords='offset points', xytext=xytext, ha='center', va=va, fontsize=10, bbox=dict(facecolor='white', boxstyle='round,pad=0.2'), zorder=104, arrowprops=dict(arrowstyle='-'))
            #else:
            plt.annotate(annote, xy, textcoords='offset points', xytext=xytext, ha=ha, va=va, fontsize=9, bbox=dict(facecolor='white', boxstyle='round,pad=0.2'), zorder=104)

        # Add Tucana B
        if distance-250 < 1400 < distance+250:
            tucB_r = 80 # pc
            tucB_mv = -6.9
            tucB_x = transform(tucB_r, x_vals, dic[x]['scale']=='log')
            tucB_y = transform(tucB_mv, y_vals, dic[y]['scale']=='log')
            plt.scatter(tucB_x, tucB_y, color='k', s=20, marker='x', zorder=101)
            xy = tucB_x, tucB_y
            xytext=[-3,3]
            ha, va = 'right', 'bottom'
            annote = 'Tuc B'
            plt.annotate(annote, xy, textcoords='offset points', xytext=xytext, ha=ha, va=va, fontsize=9, bbox=dict(facecolor='white', boxstyle='round,pad=0.2'), zorder=104)

        # Add approximate candidate location
        if distance == 2000:
            cand_x, cand_y = 3306, -8.0 # Based on values from mcmc_tight_qual_50k
            cand_x = transform(cand_x, x_vals, dic[x]['scale']=='log')
            cand_y = transform(cand_y, y_vals, dic[y]['scale']=='log')
            plt.scatter(cand_x, cand_y, marker='*', color='k', edgecolor='k', s=200, zorder=101)
            xy = cand_x, cand_y
            xytext=[-6, -4]
            ha, va = 'right', 'top'
            plt.annotate('DES J0015-3825', xy, textcoords='offset points', xytext=xytext, ha=ha, va=va, fontsize=10, zorder=104, bbox = dict(facecolor='white', boxstyle='round,pad=0.2'))
        
        # X ticks if in the last row
        if True: #ncols*nrows - idx <= ncols:
            xticks = np.arange(len(x_vals)) + 0.5
            ax.set_xticks(xticks)
            ax.set_xticklabels(map(dic[x]['conversion'], x_vals))
            plt.xlabel('{} ({})'.format(dic[x]['label'], dic[x]['unit']))

        # Y ticks if in first column:
        if True: #idx%ncols == 0:
            yticks = np.arange(len(y_vals)) + 0.5
            yticks = yticks[1::2] # Every other tick to avoid crowding the axis
            ax.set_yticks(yticks)
            yticklabels = map(dic[y]['conversion'], y_vals)
            yticklabels = yticklabels[1::2] # Every other label
            ax.set_yticklabels(yticklabels)
            plt.ylabel('{} ({})'.format(dic[y]['label'], dic[y]['unit']))

        # Add solar mass Y ticks if in last column:
        abs_mags = np.array( sorted(set(table['abs_mag']), reverse=True) )
        stellar_masses = utils.mag_to_mass(abs_mags)
        if True: #idx%ncols == ncols-1:
            yticks = np.arange(len(y_vals)) + 0.5
            yticks = yticks[1::2] # Every other tick to avoid crowding the axis
            twin_ax = ax.twinx()
            twin_ax.set_yticks(list(yticks) + [yticks[-1]+0.5])
            yticklabels = map(dic['stellar_mass']['conversion'], stellar_masses) + ['']
            twin_ax.set_yticklabels(yticklabels[1::2])
            twin_ax.set_ylabel('{} ({})'.format(dic['stellar_mass']['label'], dic['stellar_mass']['unit']))

    fig.colorbar(im, ax=axs.ravel().tolist(), label=cbar_label, pad=0.08, shrink=0.65) # Move colorbar right to make room for axis labels

    if typ == 'sigma':
        title = 'sigma_multipanel'
    elif typ == 'prob':
        title = 'prob_multipanel'
    if density_table is not None:
        title += '_blending'
    title += '_test'
    plt.savefig(title, bbox_inches='tight', dpi=200)
    plt.close()

def fourpanel(typ='prob'):
    distances = [500, 1000, 1500, 2000]
    fname = 'detection_table.fits'
    plot_sensitivity(fname, distances, 2, 2, typ=typ, density_table='blending/density_table.fits')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', '-t', type=str, default='prob')
    args = vars(parser.parse_args())

    fourpanel(args['type'])


