import loader
import predict_satellites
import astropy.io.fits as fits
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
plt.ion()
import numpy as np

pair = 'RJ'
halos = loader.Halos(pair)
sats = predict_satellites.Satellites(pair)
#sig_table = fits.open('realizations/{0}/sats_table_{0}_5trials.fits'.format(pair))[1].data
# Cut sats to match with sigma table
#distance_cut = (sats['distance'] > 300) & (sats['distance'] < 2000) #kpc
#sats = sats[distance_cut]
# Get M31 distances too
params = loader.Parameters()
Mpc_to_kpc = 1000.
x31 = params.hyper['chi'] * halos.M31['x'] * (Mpc_to_kpc/params.cosmo['h'])
y31 = params.hyper['chi'] * halos.M31['y'] * (Mpc_to_kpc/params.cosmo['h'])
z31 = params.hyper['chi'] * halos.M31['z'] * (Mpc_to_kpc/params.cosmo['h'])

M31_distance = np.sqrt((sats['x']-x31)**2 + (sats['y']-y31)**2 + (sats['z']-z31)**2)
sats['M31_distance'] = M31_distance
sats['host_distance'] = np.minimum(sats['distance'], sats['M31_distance'])

# Some plots
outdir = 'disruption_diagnostic_plots/'

# Hist of disruption probabilities
plt.hist(sats['prob'])
plt.xlabel('Survival probability')
plt.savefig(outdir+'hist')
plt.close()

plt.hist(sats['prob'][sats['host_distance'] < 2000])
plt.xlabel('Survival probability')
plt.savefig(outdir+'close_hist')
plt.close()

# Distances from MW, M31, and nearer host
# MW distance
plt.scatter(sats['distance'], sats['prob'], s=10)
plt.axvline(300, color='0.3', alpha=0.5, linestyle='--')
plt.axvline(2000, color='0.3', alpha=0.5, linestyle='--')
plt.xlabel('Distance from MW (kpc)')
plt.xlim(-100, 2100)
plt.ylabel('Survival probability')
plt.savefig(outdir+'MW_distance')
plt.close()

# M31 Distance
plt.scatter([], []) # Just to use orange instead of blue dots
plt.scatter(sats['M31_distance'], sats['prob'], s=10)
plt.axvline(300, color='0.3', alpha=0.5, linestyle='--')
plt.axvline(2000, color='0.3', alpha=0.5, linestyle='--')
plt.xlabel('Distance from M31 (kpc)')
plt.xlim(-100, 2100)
plt.ylabel('Survival probability')
plt.savefig(outdir+'M31_distance')
plt.close()

# Nearest host distance
closer_to_MW = np.where(sats['distance'] < sats['M31_distance'])[0]
closer_to_M31 = np.array([i for i in range(len(sats)) if i not in closer_to_MW])
plt.scatter(sats['host_distance'][closer_to_MW], sats['prob'][closer_to_MW], s=10, label='Closer to MW', zorder=1)
plt.scatter(sats['host_distance'][closer_to_M31], sats['prob'][closer_to_M31], s=10, label='Closer to M31', zorder=0)
plt.axvline(300, color='0.3', alpha=0.5, linestyle='--')
plt.axvline(2000, color='0.3', alpha=0.5, linestyle='--')
plt.legend(loc='lower right')
plt.xlabel('Distance from nearest host (kpc)')
plt.xlim(-100, 2100)
plt.ylabel('Survival probability')
plt.savefig(outdir+'host_distance')
plt.close()

# 2D scatter
plt.scatter(sats['distance'], sats['M31_distance'], c=sats['prob'])
plt.colorbar(label='Survival Probability')
plt.axvline(300, color='0.3', alpha=0.5, linestyle='--')
plt.axvline(2000, color='0.3', alpha=0.5, linestyle='--')
plt.xlabel('Distance from MW (kpc)')
plt.xlim(-100, 2100)
plt.ylabel('Distance from M31 (kpc)')
plt.ylim(-100, 2100)
plt.savefig(outdir+'2D_distance')
plt.close()

# 3D scatter
cut = sats['distance'] < 2100 # kpc
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
im = ax.scatter(sats[cut]['x'], sats[cut]['y'], sats[cut]['z'], s=10, c=sats[cut]['prob'])
fig.colorbar(im, label='Survival Probability')
plt.savefig(outdir+'3D_scatter')
plt.close()
"""
# 2D scatter
x0, y0, z0 = np.random.rand(3)*2000 - 1000
x1, y1, z1 = np.random.rand(3)*2000 - 1000
print(x0,y0,z0)
print(x1,y1,z1)
d0 = np.sqrt((sats['x']-x0)**2 + (sats['y']-y0)**2 + (sats['z']-z0)**2)
d1 = np.sqrt((sats['x']-x1)**2 + (sats['y']-y1)**2 + (sats['z']-z1)**2)

plt.scatter(d0, d1, c=sats['prob'])
plt.colorbar(label='Survival Probability')
#plt.axvline(300, color='0.3', alpha=0.5, linestyle='--')
#plt.axvline(2000, color='0.3', alpha=0.5, linestyle='--')
plt.xlabel('Distance from MW (kpc)')
#plt.xlim(-100, 2100)
plt.ylabel('Distance from M31 (kpc)')
#plt.ylim(-100, 2100)
#plt.savefig(outdir+'2D_distance')
#plt.close()
"""

