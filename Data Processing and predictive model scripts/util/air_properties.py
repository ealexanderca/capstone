import pandas as pd
import numpy as np

_data = pd.read_csv('./util/air_properties_ideal_gas_100kPa.csv')

t_arr = _data['T (K)']
u_arr = _data['u (kJ/kg)'] * 1000
h_arr = _data['h (kJ/kg)'] * 1000

c_v_arr = np.average(
    [np.convolve(u_arr, [0,-1,1], 'valid')/np.convolve(t_arr, [0,-1,1], 'valid'),
    np.convolve(u_arr, [1,-1,0], 'valid')/np.convolve(t_arr, [1,-1,0], 'valid')],
axis=0)

c_p_arr = np.average(
    [np.convolve(h_arr, [0,-1,1], 'valid')/np.convolve(t_arr, [0,-1,1], 'valid'),
    np.convolve(h_arr, [1,-1,0], 'valid')/np.convolve(t_arr, [1,-1,0], 'valid')],
axis=0)

t_arr = t_arr[1:-1]
u_arr = u_arr[1:-1]
h_arr = h_arr[1:-1]

gamma_arr = c_p_arr/c_v_arr

r_arr = c_p_arr - c_v_arr

R = np.average(r_arr)

def c_v(t):
    return np.interp(t, t_arr, c_v_arr)

def c_p(t):
    return np.interp(t, t_arr, c_p_arr)

def gamma(t):
    return np.interp(t, t_arr, gamma_arr)

def u(t):
    return np.interp(t, t_arr, u_arr)

def h(t):
    return np.interp(t, t_arr, h_arr)

# Should be constant
#def r(t):
#    return np.interp(t, t_arr, r_arr)

def properties(t):
    return c_v(t), c_p(t), gamma(t)#, r(t)

