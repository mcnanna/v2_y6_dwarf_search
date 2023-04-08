import matplotlib.pyplot as plt
import matplotlib
import numpy as np

plt.ion()
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

dirname = "/Users/mcnanna/Research/y6/v2_y6_dwarf_search/simulations/completeness/"


fig, axs = plt.subplots(1,2, figsize=(12., 5))
fig.subplots_adjust(wspace=0.3, bottom=0.15)

# Photoerror model

p = np.recfromcsv(dirname + 'photo_error_model.csv')
magdiff = p['mag']
log_mag_err = p['log_mag_err']
median_maglim_r = 23.94

x = magdiff + median_maglim_r
y1 = 10**log_mag_err
y2 = 0.01 + 10**log_mag_err

ax = axs[1]
ax.plot(x, y2, color='k', zorder=1)
ax.plot(x, y1, color='k', zorder=0, linestyle='dashed')
ax.set_yscale('log')
ax.set_xlim(18, 26)
ax.set_xlabel('$r$ (mag)')
ax.set_ylabel('$\sigma_r$ (mag)')

# Stellar Completeness

s = np.recfromcsv(dirname + 'y6_gold_v2_stellar_classification_summary_r_ext2.csv')
x = s['mag_r']
y1 = s['eff']
y2 = s['eff_star']

ax = axs[0]
ax.plot(x, y1, color='k', zorder=0, linestyle='dashed', label='Detection')
ax.plot(x, y2, color='k', zorder=1, label='Detection and \nClassification')
ax.set_xlim(20, 27)
ax.set_ylim(0, 1)
ax.set_xlabel('$r$ (mag)')
ax.set_ylabel('Stellar Completeness')
#ax.legend()

plt.savefig('figures/completeness.png')
#plt.close()
