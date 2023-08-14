#!/usr/env python

from astropy.io import fits
import numpy as np
import os
import paramiko
from scp import SCPClient
#from numpy.lib.recfunctions import append_fields
import scipy.interpolate
import ugali.utils.healpix
#from utils import *
#from helpers.SimulationAnalysis import readHlist

import ugali.utils.projector
from ugali.utils.projector import gal2cel, cel2gal
import ugali.utils.idl
from ugali.utils.shell import get_ugali_dir
try:
    from ugali.utils.shell import get_cat_dir
except:
    def get_cat_dir():
        return os.path.join(get_ugali_dir(),'catalogs')
from ugali.utils.logger import logger

import astropy.table
from astroquery.vizier import Vizier
Vizier.ROW_LIMIT = -1


def get_healpixel_files(cfg, nside, pixels):
    def createSSHClient(server, port, user, password):
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(server, port, user, password)
        return client

    datadir = cfg['catalog']['dirname']
    
    exists = []
    for pixel in pixels:
        if os.path.exists('{}/y6_gold_2_0_{:0>5n}.fits'.format(datadir, pixel)):
            exists.append(pixel)

    if len(exists) < len(pixels):
        print("Copying over {} files".format(len(pixels) - len(exists)))
        ssh = createSSHClient('login.hep.wisc.edu', 22, 'mcnanna', 'P!llar20')
        scp = SCPClient(ssh.get_transport())
        
        for i, pixel in enumerate(pixels):
            if pixel not in exists:
                print("File {} ...".format(i+1))
                try:
                    scp.get('~/data/skim_y6_gold_2_0/y6_gold_2_0_{:0>5n}.fits'.format(pixel), '{}/y6_gold_2_0_{:0>5n}.fits'.format(datadir, pixel))
                except Exception as e:
                    print(e)
        scp.close()
        ssh.close()


class Inputs:
    def getPhotoError(self, infile):
        d = np.recfromcsv(infile)

        x = d['mag']
        y = d['log_mag_err']

        x = np.insert(x, 0, -10.)
        y = np.insert(y, 0, y[0])

        f = scipy.interpolate.interp1d(x, y, bounds_error=False, fill_value=1.)

        return f

    def getCompleteness(self, infile):
        d = np.recfromcsv(infile)

        x = d['mag_r']
        y1 = d['eff']
        y2 = d['eff_star']

        x = np.insert(x, 0, 16.)
        y1 = np.insert(y1, 0, y1[0])
        y2 = np.insert(y2, 0, y2[0])

        f1 = scipy.interpolate.interp1d(x, y1, bounds_error=False, fill_value=0.)
        f2 = scipy.interpolate.interp1d(x, y2, bounds_error=False, fill_value=0.)

        return f1, f2

    def __init__(self, cfg):
        datadir = cfg['catalog']['dirname'] + '/../'
        cmpltdir = '/Users/mcnanna/Research/y6/v2_y6_dwarf_search/simulations/completeness/'

        self.log_photo_error = self.getPhotoError(cmpltdir + 'photo_error_model.csv')
        eff, eff_star = self.getCompleteness(cmpltdir + 'y6_gold_v2_stellar_classification_summary_r_ext2_merge.csv')
        self.efficiency = eff
        self.completeness = eff_star

        self.m_maglim_g = ugali.utils.healpix.read_map(datadir + 'y6_gold_2_0_decasu_bdf_nside4096_g_depth.fits')
        self.m_maglim_r = ugali.utils.healpix.read_map(datadir + 'y6_gold_2_0_decasu_bdf_nside4096_r_depth.fits')
        self.m_maglim_i = ugali.utils.healpix.read_map(datadir + 'y6_gold_2_0_decasu_bdf_nside4096_i_depth.fits')

        self.m_ebv = ugali.utils.healpix.read_map(datadir + 'ebv_sfd98_fullres_nside_4096_nest_equatorial.fits.gz')



class Satellites:
    def __init__(self):
        self.master = np.recfromcsv('/Users/mcnanna/Research/y3-mw-sats/data/mw_sats_master_2023.csv')
        #self.master = np.recfromcsv('/Users/mcnanna/Research/y3-mw-sats/data/mw_sats_master_extra.csv')
        self.all = self.master[np.where(self.master['type2'] >= 0)[0]]
        self.dwarfs = self.master[np.where(self.master['type2'] >= 3)[0]]
        
    def __getitem__(self, key):
        return self.master[np.where(self.master['name'] == name)[0]][0]



class SourceCatalog(object):
    DATADIR=get_cat_dir()
 
    def __init__(self, filename=None):
        columns = [('name',object),
                   ('ra',float),
                   ('dec',float),
                   ('glon',float),
                   ('glat',float),
                   ('modulus',float),
                   ('distance',float), # kpc
                   ('a_h',float), # degrees
                   ('a_physical',float), # pc
                   ('ellipticity',float),
                   ('r_h',float), # degrees
                   ('r_physical',float), # pc
                   ('m_v',float)]
        self.data = np.recarray(0,dtype=columns)
        self._load(filename)
        if np.isnan([self.data['glon'],self.data['glat']]).any():
            raise ValueError("Incompatible values")
 
    def __getitem__(self, key):
        """ 
        Support indexing, slicing and direct access.
        """
        try:
            return self.data[key]
        except ValueError as message:
            if key in self.data['name']:
                return self.data[self.data['name'] == key]
            else:
                raise ValueError(message)
 
    def __add__(self, other):
        ret = SourceCatalog()
        ret.data = np.concatenate([self.data,other.data])
        return ret
        
    def __len__(self):
        """ Return the length of the collection.
        """
        return len(self.data)
 
    def _load(self, filename):
        pass
 
    def match(self,lon,lat,coord='gal',tol=0.1,nnearest=1):
        if coord.lower() == 'cel':
            glon, glat = cel2gal(lon,lat)
        else:
            glon,glat = lon, lat
        return ugali.utils.projector.match(glon,glat,self['glon'],self['glat'],tol,nnearest)


class McConnachie15(SourceCatalog):
    """
    Catalog of nearby dwarf spheroidal galaxies. Updated September 2015.
    http://arxiv.org/abs/1204.1562

    http://www.astro.uvic.ca/~alan/Nearby_Dwarf_Database_files/NearbyGalaxies.dat
    """
    def _load(self,filename):
        if filename is None: 
            filename = os.path.join(self.DATADIR,"J_AJ_144_4/NearbyGalaxies_updated.dat")
        self.filename = filename
 
        delimiter = [19,3,3,5,3,3,3,6,6,5,5,7,5,5,5,4,4,6,5,5,5,5,5,5,4,4,7,6,6,5,5,5,5,5,5,5,5,5,5,6,5,5,6,5,5,2]
        raw = np.genfromtxt(filename,delimiter=delimiter,usecols=list(range(7))+[8,14,20,26],dtype=['|S19']+10*[float],skip_header=38)

        raw[['LMC' in name for name in raw['f0']].index(True)]['f10'] = 540.0 # LMC radius = 9 deg
        raw[['SMC' in name for name in raw['f0']].index(True)]['f10'] = 180.0 # LMC radius = 3 deg
        raw[['Bootes III' in name for name in raw['f0']].index(True)]['f10'] = 60.0 # Bootes III radius = 1 deg

        self.data.resize(len(raw))
        self.data['name'] = np.char.lstrip(np.char.strip(raw['f0']),'*')

        ra = raw[['f1','f2','f3']].view(float).reshape(len(raw),-1)
        dec = raw[['f4','f5','f6']].view(float).reshape(len(raw),-1)
        self.data['ra'] = ugali.utils.projector.hms2dec(ra)
        self.data['dec'] = ugali.utils.projector.dms2dec(dec)

        modulus = raw['f7']
        self.data['modulus'] = modulus
        distance = 10**(1+self.data['modulus']/5.) # pc
        self.data['distance'] = distance/1000.

        bad_e = 0.0
        ellipticity = raw['f9']
        ellipticity = np.array([e if e not in [99.9, 9.99] else bad_e for e in ellipticity])
        self.data['ellipticity'] = ellipticity

        a_h = raw['f10']
        # McConnachie sets "bad" radius to either 99.99 or 9.999
        bad_a = np.nan # arcmin
        a_h = np.array([a if a not in [99.99, 9.999, 999.9] else bad_a for a in a_h])
        self.data['a_h'] = a_h/60. # Half-light radius along major axis

        a_physical = np.radians(self.data['a_h'])*(self.data['distance']*1000)
        self.data['a_physical'] = a_physical

        r_h = self.data['a_h'] * np.sqrt(1-self.data['ellipticity'])
        self.data['r_h'] = r_h

        r_physical = self.data['a_physical'] * np.sqrt(1-self.data['ellipticity'])
        self.data['r_physical'] = r_physical

        #r_physical = np.radians(self.data['a_h'])*(self.data['distance']*1000) * np.sqrt(1-self.data['ellipticity'])
        #self.data['r_physical'] = r_physical

        m_v_apparent = raw['f8']
        m_v = m_v_apparent - self.data['modulus']
        self.data['m_v'] = m_v

        glon,glat = cel2gal(self.data['ra'],self.data['dec'])
        self.data['glon'],self.data['glat'] = glon,glat
