#!/usr/bin/env python

import os.path
import numpy as np
from scipy.special import erf
import pickle
from astropy.coordinates import SkyCoord
from astropy.table import Table

from helpers.SimulationAnalysis import SimulationAnalysis
import percent

import loader

def Mr_to_L(Mr, Mbol_sun=4.81):
    """Conversion from absolute magnitude to luminosity"""
    L = 10**((-1.*Mr + Mbol_sun)/2.5)
    return L


def L_to_Mr(L, Mbol_sun=4.81):
    """Conversion from luminosity to absolute magnitude"""
    Mr = -1.*(2.5*(np.log10(L))-Mbol_sun)
    return Mr


def Mr_to_mu(Mr, r12, pc_to_kpc=1000.):
    """Conversion from absolute magnitude and physical half-light radius (pc) to surface brightness"""
    mu = Mr + 36.57 + 2.5*np.log10(2*np.pi*((r12/pc_to_kpc)**2))
    return mu


class Satellites:
    """
    Assigns satellites to DM halos
    """
    def __init__(self,pair,interpolator='interpolator.pkl'):
        self.halos = loader.Halos(pair) # Halo coordinates are translated such that the Milky Way is at the origin
        params = loader.Parameters()
        with open(interpolator, 'rb') as interp:
            try:
                vpeak_Mr_interp = pickle.load(interp)
            except:
                vpeak_Mr_interp = pickle.load(interp, encoding='latin1')

        # Cut subhalo catalog
        self.subhalos = self.cut_subhalo_catalog(self.halos.subhalos, params)

        # Check if both accretion-time radii and pericenter have been calculated
        accretion_fname = 'sims/{0}/{0}_accretion_rvir.npy'.format(pair)
        pericenter_fname = 'sims/{0}/{0}_perihelion.npy'.format(pair)
        self.rvir_acc, self.rs_acc, self.d_peri, self.a_peri = self.tree_calculations(self.halos, self.subhalos, accretion_fname=accretion_fname, pericenter_fname=pericenter_fname)

        # Calculate M_r luminosities from V_peak
        self.M_r = self.vpeak_to_Mr(self.subhalos['vpeak'], params, vpeak_Mr_interp)

        # Calculate distance from MW and (x,y,z) coordinates in kpc
        self.distance, self.position = self.get_halocentric_positions(self.halos, self.subhalos, params)

        # Calculate sizes (in pc) and surface brightness
        self.r_physical = self.get_sizes(self.rvir_acc, self.rs_acc, self.subhalos, params)
        self.mu = Mr_to_mu(self.M_r, self.r_physical)

        # Calculate disruption probability due to baryonic effects and occupation fraction (no WDM suppression)
        # TODO
        ML_prob = np.zeros(len(self.subhalos))
        self.prob = self.get_survival_prob(ML_prob, self.subhalos['mpeak'], params)

        # Gather into nice table
        self.table = Table()
        self.table['M_r'] = self.M_r
        self.table['distance'] = self.distance
        self.table['x'], self.table['y'], self.table['z'] = self.position.T
        self.table['r_physical'] = self.r_physical
        self.table['mu'] = self.mu
        self.table['prob'] = self.prob

    def __len__(self):
        return len(self.table)
    def __getitem__(self, idx):
        return self.table[idx]
    def __setitem__(self, idx, val):
        self.table[idx] = val

    def ra_dec(self, psi=0):
        transform = self.halos.rotation(psi)
        xp, yp, zp = np.dot(transform, self.position.T)
        sky_coord = SkyCoord(x=xp, y=yp, z=zp, representation='cartesian', unit='kpc').spherical
        ra, dec = sky_coord.lon.degree, sky_coord.lat.degree
        return ra, dec

    # All of these fucntion don't really need to be in the class
    # They're written so as to be easy to apply more generally
    def cut_subhalo_catalog(self, subhalos, params):
        """Applies resolution cuts to subhalo catalog"""
        mpeak_cut = (subhalos['mpeak']*(1.-params.cosmo['omega_b']/params.cosmo['omega_m']) > params.hyper['mpeak_cut'])
        vpeak_cut = (subhalos['vpeak'] > params.hyper['vpeak_cut'])
        vmax_cut = (subhalos['vmax'] > params.hyper['vmax_cut'])
        cut_idx = mpeak_cut & vpeak_cut & vmax_cut
        return subhalos[cut_idx]

    def tree_calculations(self, halos, subhalos, accretion_fname='accretion_rvir.npy', pericenter_fname='pericenter.npy', Mpc_to_kpc=1000.):
        """
        Calculates accretion-time r_vir and rs, and pericenter distance and scale factor.
        All distances in kpc.
        Note: pericenter distance is a straightforward pythagoreon calculation from the 
        halo coordinates. Any modification from expansion must be applied elsewhere.
        """
        if os.path.isfile(accretion_fname):
            rvir_acc, rs_acc = np.load(accretion_fname)
            need_acc = False
        else:
            need_acc = True
        if os.path.isfile(pericenter_fname):
            d_peri, a_peri = np.load(pericenter_fname)
            need_peri = False
        else:
            need_peri = True

        if not (need_acc or need_peri):
            return rvir_acc, rs_acc, d_peri, a_peri

        print("{} {} not yet calculated".format("Accretion" if need_acc else "", "Pericenter" if need_peri else""))
        sim = SimulationAnalysis(trees_dir = 'sims/{}/trees'.format(halos.pair))
        print("Loading M31 and MW main branches...")
        tree_M31 = sim.load_main_branch(halos.M31['id'])
        tree_MW = sim.load_main_branch(halos.MW['id'])

        main_branches = []
        print("Loading {} subhalo branches...".format(len(subhalos)))
        count = 0
        for subhalo in subhalos:
            branch = sim.load_main_branch(subhalo['id'])
            main_branches.append(branch)
            count += 1
            percent.bar(count, len(subhalos))
        
        if need_acc:
            rvir_acc = np.zeros(len(main_branches))
            rs_acc = np.zeros(len(main_branches))
            for i in range(len(main_branches)):
                branch = main_branches[i]
                # Check if halo becomes M31 or MW subhalo
                M31_overlap = np.isin(branch['upid'], tree_M31['id'])
                MW_overlap = np.isin(branch['upid'], tree_MW['id'])
                if True in M31_overlap:
                    rvir_acc[i] = branch[M31_overlap]['rvir'][-1]
                    rs_acc[i] = branch[M31_overlap]['rs'][-1]
                elif True in MW_overlap:
                    rvir_acc[i] = branch[MW_overlap]['rvir'][-1]
                    rs_acc[i] = branch[MW_overlap]['rs'][-1]
                else:
                    peak_idx = np.argmax(branch['rvir'])
                    rvir_acc[i] = branch['rvir'][peak_idx]
                    rs_acc[i] = branch['rs'][peak_idx]
            np.save(accretion_fname, (rvir_acc, rs_acc))

        if need_peri:
            d_peri = np.zeros(len(main_branches))
            a_peri = np.zeros(len(main_branches))
            for i in range(len(main_branches)):
                branch = main_branches[i]
                n = len(branch['scale']) # Number of time steps for which the branch has existed
                MW_x, MW_y, MW_z = tree_MW['x'][:n], tree_MW['y'][:n], tree_MW['z'][:n]
                M31_x, M31_y, M31_z = tree_M31['x'][:n], tree_M31['y'][:n], tree_M31['z'][:n]
                x, y, z = branch['x'], branch['y'], branch['z']
                
                d_MW = np.sqrt((x-MW_x)**2 + (y-MW_y)**2 + (z-MW_z)**2)
                d_M31 = np.sqrt((x-M31_x)**2 + (y-M31_y)**2 + (z-MW_z)**2)
                MW_peri = np.min(d_MW)
                M31_peri = np.min(d_M31)
                
                if MW_peri < M31_peri:
                    d_peri[i] = MW_peri*Mpc_to_kpc
                    idx = np.argmin(d_MW) # faster to use np.where(MW_peri==d_peri)?
                else:
                    d_peri[i] = M31_peri*Mpc_to_kpc
                    idx = np.argmin(d_M31)
                a_peri[i] = branch['scale'][idx]
            np.save(pericenter_fname, (d_peri, a_peri))

        return rvir_acc, rs_acc, d_peri, a_peri

    def vpeak_to_Mr(self, vpeak, params, interpolator):
        """Converts subhalo V_peak to absolute r-band magnitude"""
        sort_idx = np.argsort(np.argsort(vpeak))
        # The interpolater will return a sorted list of Mr, so we need to re-sort it 
        # to match the original ordering of subhalos
        Mr_mean = interpolator(vpeak, params.connection['alpha'])[sort_idx]
        L_mean = Mr_to_L(Mr_mean)
        # Draw luminosity from lognormal distribution:
        L = np.random.lognormal(np.log(L_mean), np.log(10)*params.connection['sigma_M'])
        Mr = L_to_Mr(L)
        return Mr

    def get_halocentric_positions(self, halos, subhalos, params, Mpc_to_kpc=1000.):
        """Gets halocentric radii and Cartesian coordinates of subhalos in kpc"""
        x = params.hyper['chi'] * subhalos['x'] * (Mpc_to_kpc/params.cosmo['h'])
        y = params.hyper['chi'] * subhalos['y'] * (Mpc_to_kpc/params.cosmo['h'])
        z = params.hyper['chi'] * subhalos['z'] * (Mpc_to_kpc/params.cosmo['h'])
        radii = np.sqrt(x**2 + y**2 + z**2)#*params.hyper['chi']
        position = np.vstack((x,y,z)).T
        return radii, position

    def get_sizes(self, rvir_acc, rs_acc, subhalos, params, c_normalization=10.0):
        """
        Gets half-light radii of satellites in pc
        Args:
            rvir_acc (array of floats): virial radii of subhalos at accretion in pc
            rs_acc (array of floats): NFW scale radii of subhalos at accretion in pc
            c_normalization (float): concentration normalization in Jiang+ size relation
        """
        c_acc = rvir_acc/rs_acc
        c_correction = (c_acc/c_normalization)**params.hyper['gamma_r']
        beta_correction = ((subhalos['vmax']/subhalos['vacc']).clip(max=1.0))**params.hyper['beta']
        halo_r12 = params.connection['A']*c_correction*beta_correction * ((rvir_acc/params.hyper['R0']*params.cosmo['h']))**params.connection['n']
        r12 = np.random.lognormal(np.log(halo_r12),np.log(10)*params.connection['sigma_r'])
        return r12

    def occupation_fraction(self, mpeak, params):
        """Gets satellite occupation fraction """
        fgal = (0.5*(1.+erf((np.log10((mpeak/params.cosmo['h'])*(1-params.cosmo['omega_b']/params.cosmo['omega_m']))-params.connection['M50'])/(np.sqrt(2)*params.connection['sigma_mpeak'])))).clip(max=1.)
        return fgal

    def get_survival_prob(self, ML_prob, mpeak, params):
        """Gets satellite survival probabiliy from random forest ML classifier data. No WDM suppression"""
        baryonic_disruption_prob = ML_prob**(1./params.connection['B'])
        occupation_prob = self.occupation_fraction(mpeak, params)
        prob = (1.-baryonic_disruption_prob)*occupation_prob
        return prob

