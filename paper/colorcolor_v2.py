import sys
import numpy as np

def sel(survey, data):
    m, b = 0.369485, -0.0555077
    def distance(x,y): # Distance from isochrone line in color-color space
        return np.abs(y-(m*x+b)) / np.sqrt(m**2+1) 
    x = data[survey.mag_1] - data[survey.mag_2] # g-r
    y = data[survey.mag_2] - data[survey.mag_3] # r-i
    color_tol = 0.15
    return (distance(x, y) < np.sqrt(color_tol**2 + data[survey.mag_err_1]**2 + data[survey.mag_err_2]**2 + data[survey.mag_err_3]**2))

