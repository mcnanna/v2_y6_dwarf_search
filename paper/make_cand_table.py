import yaml
import numpy as np
from collections import OrderedDict

base = """\\begin{deluxetable}{l c c }
\\tabletypesize{\\footnotesize}
\\tablecaption{\label{tab:properties} Candidate Galaxy Properties}
\\tablehead{\colhead{Parameter} & \colhead{Value} & \colhead{Units}}
\startdata
"""

#               [title, decimals, unit, factor(optional), reverse(optional)]
quants = OrderedDict()
quants['ra'] = ['\\ra', 3, 'deg']
quants['dec'] = ['\dec', 3, 'deg']
quants['distance_modulus'] = ['\mM', 2, '\magn']
quants['distance'] = ['Distance', 2, 'Mpc', 1./1000]
quants['extension'] = ['\major', 1, 'arcmin', 60]
quants['extension_radial'] = ['\\rhalf', 1, 'arcmin', 60]
quants['physical_size_radial'] = ['$r_{1/2}$', 1, 'kpc']
quants['ellipticity'] = ['\ellip', 2, '\ldots']
quants['position_angle_cel'] = ['P.A.', 0, 'deg']
quants['age'] = ['\\tau', 1, 'Gyr']
quants['metallicity'] = ['$Z$', 1, '\ldots', 1e4] # Special case with 10^{-4} after value
#\hline
quants['Mv'] = ['$M_V$', 1, '\magn', 1., True]
quants['luminosity'] = ['$L_V$', 1, '10^5 \Lsolar', 1e-5]
quants['surface_brightness'] = ['$\mu_0$', 1, '\magn arcsec^{-2}', 1., True] # Special case with no errors
quants['mass'] = ['$M_*$', 2, '10^5 \Msolar', 1e-5]
quants['feh'] = ['{[Fe/H]}', 1, 'dex'] # Special case with no errors
quants['glon'] = ['$l$', 3, 'deg']
quants['glat'] = ['$b$', 3, 'deg']


end = """\enddata
{\\footnotesize \\tablecomments{Improve title. PA is cel}}
\end{deluxetable}
"""


def errs(name):  
    res = y['results'][name]  
    c = res[0]  
    m = res[1][0]  
    p = res[1][1]  
    #print(c, ' +',p-c, ' -',c-m)  
    return np.array([c, p-c, c-m])  

with open('/Users/mcnanna/Research/ngc55/mcmc/mcmc_tight_qual_50k/ngc55_cand_mcmc.yaml') as ymlfile:
    y = yaml.load(ymlfile)


with open('table_properties.tex', 'w') as t:
    t.write(base)

    for q in quants.keys():
        if q == 'Mv':
            t.write('\hline \n')

        info = quants[q]
        title, figs, unit = info[:3]
        reverse=False
        factor=1.
        if len(info) == 5:
            reverse = info[4]
        if len(info) >= 4:
            factor = info[3]

        if q in ['surface_brightness', 'feh']:
            try:
                res = y['results'][q][0]
            except TypeError:
                res = y['results'][q]
            if figs==0:
                val = int(factor*res)
            else:
                val = round(factor*res, figs)
            line = "{title} & {val} & {unit} \\\ \n".format(title=title, val=round(val,figs)*factor, unit=unit)
            t.write(line)

        else:
            val_array = factor*errs(q)
            if figs == 0:
                val_array = val_array.astype(int)
            else:
                val_array = val_array.round(figs).astype(str)
                for i,v in enumerate(val_array):
                    if len(v) < figs+2: # 2 extra for the leading number and the decimal
                        val_array[i] += '0'

            val, ep, em = val_array
            if reverse:
                ep, em = em[1:], ep[1:] # Reverse and removes leading - sign

            if q == 'metallicity':
                line = "{title} & {val} (\substack{{+{ep}\\\-{em}}}) \\times 10^{{-4}} & {unit} \\\ \n".format(title=title, val=val, ep=ep, em=em, unit=unit)
            else:
                line = "{title} & {val} \substack{{+{ep}\\\-{em}}} & {unit} \\\ \n".format(title=title, val=val, ep=ep, em=em, unit=unit)
            t.write(line)

    t.write(end)

