import yaml
import numpy as np

base = """\\begin{deluxetable}{l c c }
\\tabletypesize{\\footnotesize}
\\tablecaption{\label{table:properties} Candidate Galaxy Properties}
\\tablehead{\colhead{Parameter} & \colhead{Value} & \colhead{Units}}
\startdata\n
"""

#               [title, unit, factor(optional), reverse(optional)]
quants = {'ra': ['\\ra', 'deg'],
          'dec': ['\dec', 'deg'],
          'distance_modulus': ['\mM', '\magn'],
          'distance': ['Distance', 'Mpc', 1./1000],
          'extension': ['\major', 'arcmin', 1./60],
          'extension_radial': ['\\rhalf', 'arcmin', 1/60.],

print(base)


def errs(name):  
    res = y['results'][name]  
    c = res[0]  
    m = res[1][0]  
    p = res[1][1]  
    print(c, ' +',p-c, ' -',c-m)  
    return np.array([c, p-c, c-m])  
