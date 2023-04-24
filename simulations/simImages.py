import numpy as np
import numpy.ma as ma
import astropy.io.fits as fits
from astropy.table import Table
from astropy.visualization import make_lupton_rgb
from photutils.datasets import make_gaussian_sources_image
import matplotlib.pyplot as plt
plt.ion()
import ugali.utils.projector

pix_to_arcsec = 0.236
pix_to_deg = pix_to_arcsec/3600
arcsec_to_pix = 1/pix_to_arcsec
deg_to_pix = 1/pix_to_deg

def sdss_rgb(imgs, bands, scales=None, m = 0.02): # Stolen from https://github.com/legacysurvey/imagine/blob/57ec68ea447f56b9265b6645427d77cd76a8b327/map/views.py
    rgbscales = {'u': (2,1.5), #1.0,
                 'g': (2,2.5),
                 'r': (1,1.5),
                 'i': (0,1.0),
                 'z': (0,0.4), #0.3
                 }
    if scales is not None:
        rgbscales.update(scales)

    I = 0
    for img,band in zip(imgs, bands):
        plane,scale = rgbscales[band]
        img = np.maximum(0, img * scale + m)
        I = I + img
    I /= len(bands)
        
    # b,g,r = [rimg * rgbscales[b] for rimg,b in zip(imgs, bands)]
    # r = np.maximum(0, r + m)
    # g = np.maximum(0, g + m)
    # b = np.maximum(0, b + m)
    # I = (r+g+b)/3.
    Q = 20
    fI = np.arcsinh(Q * I) / np.sqrt(Q)
    I += (I == 0.) * 1e-6
    H,W = I.shape
    rgb = np.zeros((H,W,3), np.float32)
    for img,band in zip(imgs, bands):
        plane,scale = rgbscales[band]
        rgb[:,:,plane] = (img * scale + m) * fI / I

    # R = fI * r / I
    # G = fI * g / I
    # B = fI * b / I
    # # maxrgb = reduce(np.maximum, [R,G,B])
    # # J = (maxrgb > 1.)
    # # R[J] = R[J]/maxrgb[J]
    # # G[J] = G[J]/maxrgb[J]
    # # B[J] = B[J]/maxrgb[J]
    # rgb = np.dstack((R,G,B))
    rgb = np.clip(rgb, 0, 1)
    return rgb

def dr2_rgb(rimgs, bands, **ignored):
    return sdss_rgb(rimgs, bands, scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)


def get_rgb(imgs, bands, **kwargs):
        m=0.03
        Q=20
        mnmx=None
        clip=True
        allbands = ['g','r','i','z']
        rgb_stretch_factor = 1.5
        rgbscales=dict(
            g =    (2, 6.0 * rgb_stretch_factor),
            r =    (1, 3.4 * rgb_stretch_factor),
            i =    (0, 3.0 * rgb_stretch_factor),
            z =    (0, 2.2 * rgb_stretch_factor),
            )
        I = 0
        for img,band in zip(imgs, bands):
            plane,scale = rgbscales[band]
            img = np.maximum(0, img * scale + m)
            I = I + img
        I /= len(bands)
        if Q is not None:
            fI = np.arcsinh(Q * I) / np.sqrt(Q)
            I += (I == 0.) * 1e-6
            I = fI / I
        H,W = I.shape
        rgb = np.zeros((H,W,3), np.float32)

        rgbvec = dict(
            g = (0.,   0.,  0.75),
            r = (0.,   0.5, 0.25),
            i = (0.25, 0.5, 0.),
            z = (0.75, 0.,  0.))

        for img,band in zip(imgs, bands):
            _,scale = rgbscales[band]
            rf,gf,bf = rgbvec[band]
            if mnmx is None:
                v = (img * scale + m) * I
            else:
                mn,mx = mnmx
                v = ((img * scale + m) - mn) / (mx - mn)
            if clip:
                v = np.clip(v, 0, 1)
            if rf != 0.:
                rgb[:,:,0] += rf*v
            if gf != 0.:
                rgb[:,:,1] += gf*v
            if bf != 0.:
                rgb[:,:,2] += bf*v
        return rgb


class Image:
    def __init__(self, image, origin=None, center=None):
        """Single-band image stored as a 2D numpy array.
        
        All images have 0.236 arcsec/pixel

        center: (RA, DEC) at the image center
        origin: (RA, DEC) at the image origin, the [0, 0] entry (i.e. upper left)
            This should be the MAXIMUM RA and MINIMUM DEC. 

        When plotted with plt.imshow(origin='lower', RA will increase to the left
            and DEC will increase upwards.
        """

        if (center is None) and (origin is None):
            raise ValueError("One of center or origin must be specified")

        self.image = image
        self.shape = self.image.shape
        self.size = self.image.size
        
        ra_pix, dec_pix = self.shape
        if origin is not None:
            self.origin = origin
            self.ra_origin, self.dec_origin = origin
            self.ra_center = self.ra_origin - ra_pix/2.*pix_to_deg
            self.dec_center = self.dec_origin + dec_pix/2.*pix_to_deg
            self.center = (self.ra_center, self.dec_center)
        elif center is not None:
            self.center = center
            self.ra_center, self.dec_center = center
            self.ra_origin = self.ra_center + ra_pix/2.*pix_to_deg
            self.dec_origin = self.dec_center + dec_pix/2.*pix_to_deg
            self.origin = (self.ra_origin, self.dec_origin)

        self.ra_max = self.ra_origin
        self.ra_min = self.ra_max - ra_pix*pix_to_deg
        self.dec_min = self.dec_origin
        self.dec_max = self.dec_min + dec_pix*pix_to_deg

    def __add__(self, im):
        ## Assumes one image is fully contained within the other
        small, big = sorted([self,im], key=lambda i: i.image.size)
        
        # Find where the smaller image sits within the larger image
        
        #small_ra_pix_start = int((big.ra_origin - small.ra_origin)*deg_to_pix)
        #small_dec_pix_start = int((small.dec_origin - big.dec_origin)*deg_to_pix)
        small_ra_pix_start = abs(int((big.ra_origin - small.ra_origin)*deg_to_pix))
        small_dec_pix_start = abs(int((small.dec_origin - big.dec_origin)*deg_to_pix))

        """
        ### Option 1: Insert small array into big array
        # Create zero array the same shape as big, then insert small array
        enlarged = np.zeros(big.shape)
        enlarged[small_dec_pix_start:small_dec_pix_start+small.shape[0], small_ra_pix_start:small_ra_pix_start+small.shape[1]] = small.image
        added = Image(image = big.image+enlarged, origin=big.origin)
        """

        ### Option 2: Slice relevant piece of big array, add small array
        sliced = big.image[small_dec_pix_start:small_dec_pix_start+small.shape[0], small_ra_pix_start:small_ra_pix_start+small.shape[1]]
        added = Image(image = sliced+small.image, origin=small.origin)

        return added

    def show(self, **kwargs):
        plt.imshow(self.image, origin='lower', **kwargs)

        
class ImageFile(Image):
    def __init__(self, filename):
        d = fits.open(filename)[1]
        header = d.header
        origin = (header['RAC1'], header['DECC1'])
        image = d.data
        Image.__init__(self, image, origin=origin)


class RGBImage:
    def __init__(self, imR, imG, imB):
        self.imR = imR
        self.imG = imG
        self.imB = imB
        # Pull coordinates, etc from one of the images
        pull = ['ra', 'dec', 'center', 'origin']
        for attr_name in vars(imR):
            if any([st in attr_name for st in pull]):
                setattr(self, attr_name, getattr(imR, attr_name))
        """
        self.origin = imR.origin
        self.ra_origin, self.dec_origin = self.origin
        self.center = imR.center
        self.ra_center, self.dec_center = self.center
        self.ra_max = self.ra_origin
        self.ra_min = imR.ra_min
        self.dec_min = self.dec_origin
        self.dec_max = imR.dec_min
        """        
        self.image = np.stack((imR.image, imG.image, imB.image), axis=2) # NxNx3 array. Matches output shape of make_lupton_rgb
        self.shape = self.image.shape
        self.size = self.image.size

    def __add__(self, im2):
        sumR = self.imR + im2.imR
        sumG = self.imG + im2.imG
        sumB = self.imB + im2.imB
        return RGBImage(sumR, sumG, sumB)

    def show(self, Q=3., stretch=23., **kwargs):
        display = make_lupton_rgb(self.imR.image, self.imG.image, self.imB.image, Q=Q, stretch=stretch)
        plt.imshow(display, origin='lower', **kwargs)




def create_sat_image(sat, band, fwhm, pixel_scale=0.263, zeropoint=30., cut=None):
    """ fwhm in arcsec. 
    pixel_scale = arcsec/pixel """
    sigma = fwhm/(2*np.sqrt(2*np.log(2))) * (1./pixel_scale)

    # Cut star catalog at least to zeropoint
    if cut is None:
        cut = sat.stars['PSF_MAG_{}_CORRECTED'.format(band.upper())] < zeropoint
        sat.stars = sat.stars[cut]

    mags = sat.stars['PSF_MAG_{}_CORRECTED'.format(band.upper())]
    fluxes = 10**((mags-zeropoint)/-2.5)

    ra_origin = max(sat.stars['RA'])
    dec_origin = min(sat.stars['DEC'])

    x = (ra_origin - sat.stars['RA'])*deg_to_pix
    y = (sat.stars['DEC'] - dec_origin)*deg_to_pix

    sources = Table()
    sources['flux'] = fluxes
    sources['x_mean'] = x
    sources['y_mean'] = y
    sources['x_stddev'] = sigma * np.ones(len(sat.stars))
    sources['y_stddev'] = sigma * np.ones(len(sat.stars))
    shape = (int(max(y)+10*sigma)+1, int(max(x)+10*sigma)+1)
    # No noise for now
    image = make_gaussian_sources_image(shape, sources)
    return Image(image, origin=(ra_origin, dec_origin))

def create_rgb_sat_image(sat, fwhm, pixel_scale=0.263, zeropoint=30.):
    """ fwhm in arcsec. 
    pixel_scale = arcsec/pixel """
    cut = np.tile(True, len(sat.stars))
    for band in 'irg':
        cut &= (sat.stars['PSF_MAG_{}_CORRECTED'.format(band.upper())] < zeropoint)
    sat.stars = sat.stars[cut]
    print("{} stars in image ".format(len(sat.stars)))
    irg = [create_sat_image(sat, band, fwhm, pixel_scale, zeropoint, cut) for band in 'irg']
    return RGBImage(*irg)

if __name__ == "__main__":
    import yaml
    import load_data
    import simSatellite

    # Field
    print("Creating field images for each band...")
    g = ImageFile('/Users/mcnanna/Downloads/DES0017-3832_r4907p01_g_nobkg.fits.fz')
    r = ImageFile('/Users/mcnanna/Downloads/DES0017-3832_r4907p01_r_nobkg.fits.fz')
    i = ImageFile('/Users/mcnanna/Downloads/DES0017-3832_r4907p01_i_nobkg.fits.fz')
    z = ImageFile('/Users/mcnanna/Downloads/DES0017-3832_r4907p01_z_nobkg.fits.fz')


    print("Creating RBG field image...")
    field = RGBImage(i,r,g)
    #print("Plotting field...")
    #plt.figure()
    #field.show()
    #plt.savefig('image_plots/field.png')
    #plt.close()

    print("Loading inputs...")
    with open('../local_v2.yaml', 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    inputs = load_data.Inputs(cfg)

    ra, dec = 4.82, -38.44
    
    """
    # TucB parameters
    distance = 1400.
    abs_mag = -6.9
    r_physical = 80.
    ellipticity=0.1
    tucb_sat = simSatellite.SimSatellite(inputs, ra, dec, distance, abs_mag, r_physical, ellipticity, use_completeness=False)
    
    print("Creating TucB simulation image...")
    fwhm = 1.0
    tucb = create_rgb_sat_image(tucb_sat, fwhm)
    print("Ploting TucB...")
    plt.figure()
    tucb.show()
    plt.savefig("image_plots/tucb_{}.png".format(fwhm))
    plt.close()

    print("Adding TucB to field and plotting")
    added = field+tucb
    plt.figure()
    added.show()
    plt.savefig("image_plots/tucb_added_{}.png".format(fwhm))
    plt.close()
    """
    # NGC55 candidate paramaters
    distance = 2200. #kpc
    abs_mag = -7.9
    r_physical = 2200 #pc
    ellipticity = 0.55
    ngc55_sat = simSatellite.SimSatellite(inputs, ra, dec, distance, abs_mag, r_physical, ellipticity, use_completeness=True)
    # Make an aperature cut to avoid having such a large spatial extent
    angsep = ugali.utils.projector.angsep(ra, dec, ngc55_sat.stars['RA'], ngc55_sat.stars['DEC'])
    angsep_sel = angsep < 10.0/60. # deg, SOF aperature was 6.6', so this should be plenty
    print(np.count_nonzero(angsep_sel))
    ngc55_sat.stars = ngc55_sat.stars[angsep_sel]
            

    print("Creating NGC55 cand image...")
    fwhm = 1.0
    ngc55 = create_rgb_sat_image(ngc55_sat, fwhm)
    print("Ploting NGC55 cand...")
    plt.figure()
    ngc55.show()
    plt.savefig("image_plots/ngc55_{}.png".format(fwhm))
    plt.close()

    print("Adding NGC55 cand to field and plotting")
    added = field+ngc55
    plt.figure()
    added.show()
    plt.savefig("image_plots/ngc55_added_{}.png".format(fwhm))
    plt.close()
    
    
