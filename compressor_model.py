from util import air_properties as air
from util.util import moving_average

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from time import perf_counter
import cProfile

R_U = 8.31446261815324*1000  # J/K/kmol
PSI_TO_PA = 6894.76
ATM_PRESS = 101325  # Pa
ATM_TEMP = 20 + 273.15  # K (20degC is reference for Specific Gravity)

# TODO:
#PISTON_ROD_LENGTH =
TDC_DISTANCE = 0.0025
BDC_DISTANCE = TDC_DISTANCE + 0.012
PISTON_DIAMETER = 0.04
#CYLINDER_HEAT_COEFF =

OUTLET_VOLUME = 0.017 * np.pi * 0.054**2 / 4
REGULATOR_PRESSURE = 57 * PSI_TO_PA + ATM_PRESS  # Pa
REGULATOR_KV = (21.5/1000/60) * np.sqrt(1/(57 * PSI_TO_PA)) * 0.4
DISCHARGE_COEFF = 0.7
DIAMETER_RATIO = 0
EXPANSIBILITY_FACTOR = 1
OUTLET_VALVE_A = 0.5 * 6 * np.pi/4*0.002**2
INLET_VALVE_A = 0.3 * np.pi/4*0.0065**2
#SMOOTH_ZONE = 1  # Pa

# TODO:
MOTOR_POLES = 4
MOTOR_R1TH = 23.83  # Ohms
MOTOR_R2 = 18.96  # Ohms
MOTOR_XTH_TOTAL = 28.71  # Ohms
r,l,rho = 0.015, 0.2, 8000
ROTOR_INTERTIA =1/2 * (np.pi*r**2*l*rho) * r**2
#TOTAL_FRICTION =
MAINS_V = 120  # V (RMS)
MAINS_F = 60  # Hz

END_TIME = 1.0
MAX_SOLVER_STEP = 1E-4
SAMPLE_TIME = 3.3E-5
EQM_TOL = 1E-3

crank_r = (BDC_DISTANCE - TDC_DISTANCE)/2
piston_a = np.pi * PISTON_DIAMETER**2 / 4
rho_atm = ATM_PRESS/air.R/ATM_TEMP
h_atm = air.h(ATM_TEMP)
air_critical_pr = (2/(air.gamma(ATM_TEMP) + 1)) ** (air.gamma(ATM_TEMP) / (air.gamma(ATM_TEMP) - 1))

#T_LOW = 200
#T_HIGH = 500
#piston_cvb = air.c_v(T_LOW)
#piston_cvm = (air.c_v(T_HIGH) - piston_cvb)/(T_HIGH - T_LOW)
#piston_hb = air.h(T_LOW)
#piston_hm = (air.h(T_HIGH) - piston_hb)/(T_HIGH - T_LOW)

orifice_mass_flow_coeff = DISCHARGE_COEFF/np.sqrt(1-DIAMETER_RATIO) * EXPANSIBILITY_FACTOR * np.sqrt(2)
def orifice_mass_flow(A, rho, p_upstream, p_downstream):
    p_diff = (p_upstream > p_downstream) * (p_upstream - p_downstream)

    dmdt = orifice_mass_flow_coeff * A * np.sqrt(p_diff*rho)

    return dmdt

def kv_mass_flow(kv, rho, p_upstream, p_downstream):
    p_upstream_unchocked = np.minimum(p_upstream, p_downstream/air_critical_pr)

    sg = rho / rho_atm

    dmdt_unchoked = kv * rho * np.sqrt((p_upstream_unchocked - p_downstream)/sg)
    dmdt_choked = np.maximum(0, dmdt_unchoked / p_upstream_unchocked * p_upstream - dmdt_unchoked)

    dmdt = (p_upstream > p_downstream) * (dmdt_unchoked + dmdt_choked)

    return dmdt


def step(t, y, piston_in_leak_ratio, piston_out_leak_ratio):
    piston_m, piston_t, theta, d_theta_dt, outlet_m, outlet_t = y

    x = (TDC_DISTANCE + BDC_DISTANCE) / 2 - crank_r * np.cos(theta)
    d_x_dt = crank_r * np.sin(theta) * d_theta_dt

    piston_v = x * piston_a
    d_piston_v_dt = d_x_dt * piston_a

    # Piston fluid properties
    piston_p = piston_m * air.R * piston_t / piston_v
    piston_rho = piston_m / piston_v
    #piston_cv = piston_cvm * piston_t + piston_cvb
    #piston_h = piston_hm * piston_t + piston_hb
    piston_cv = air.c_v(piston_t)
    piston_h = air.h(piston_t)

    # Outlet cavity fluid properties
    outlet_p = outlet_m * air.R * outlet_t / OUTLET_VOLUME
    outlet_rho = outlet_m / OUTLET_VOLUME
    # TODO: Interpolate vectorized with piston properties?
    outlet_cv = air.c_v(outlet_t)
    outlet_h = air.h(outlet_t)

    # Piston valve flow
    dmdt_piston_in = orifice_mass_flow(INLET_VALVE_A, rho_atm, ATM_PRESS, piston_p)#*(1-np.exp(-((piston_p - ATM_PRESS)/SMOOTH_ZONE)**2))
    dmdt_piston_in_leak = orifice_mass_flow(INLET_VALVE_A*piston_in_leak_ratio, rho_atm, piston_p, ATM_PRESS)  # Opposed to usual in direction
    dmdt_piston_in -= dmdt_piston_in_leak
    dmdt_piston_out = orifice_mass_flow(OUTLET_VALVE_A, piston_rho, piston_p, outlet_p)#*(1-np.exp(-((piston_p - ATM_PRESS)/SMOOTH_ZONE)**2))
    dmdt_piston_out_leak = orifice_mass_flow(OUTLET_VALVE_A*piston_out_leak_ratio, piston_rho, outlet_p, piston_p)  # Opposed to usual out direction
    dmdt_piston_out -= dmdt_piston_out_leak
    d_piston_m_dt = (dmdt_piston_in - dmdt_piston_out)  # *(1-np.exp(-((piston_p - ATM_PRESS)/SMOOTH_ZONE)**2))

    # Piston temperature
    d_piston_t_dt = 1/piston_m/piston_cv * (h_atm * dmdt_piston_in - piston_h * dmdt_piston_out - piston_p * d_piston_v_dt)

    # Outlet regulator flow
    reg_p = np.minimum(REGULATOR_PRESSURE, outlet_p)
    dmdt_outlet_in = dmdt_piston_out
    dmdt_outlet_out = kv_mass_flow(REGULATOR_KV, outlet_rho, reg_p, ATM_PRESS)
    d_outlet_m_dt = dmdt_outlet_in - dmdt_outlet_out

    # Outlet cavity temperature
    d_outlet_t_dt = 1 / outlet_m / outlet_cv * (piston_h * dmdt_outlet_in - outlet_h * dmdt_outlet_out)


    y = crank_r*np.sin(theta)

    omega_e = 2*np.pi * MAINS_F * 2 / MOTOR_POLES
    s = (omega_e - d_theta_dt)/omega_e
    s += (s == 0)  # Fast method to guard against undefined torque at s == 0
    # TODO: Something is wrong with this equation
    #I_2 = np.nan_to_num(MAINS_V/((MOTOR_R1TH + MOTOR_R2/s)**2 + (MOTOR_XTH_TOTAL)**2))
    I_2 = 0
    #T_m = np.nan_to_num(3*MOTOR_POLES/2 * I_2**2 * MOTOR_R2/omega_e/s)
    T_m = 3 * MOTOR_POLES / 2 * MAINS_V ** 2 / omega_e * (MOTOR_R2 / s) / ((MOTOR_R1TH + MOTOR_R2 / s) ** 2 + (MOTOR_XTH_TOTAL) ** 2)
    T_m *= (s != 0)  # Fast method to guard against undefined torque at s == 0

    d_dthetadt_dt = 1 / ROTOR_INTERTIA * (T_m + (piston_p - ATM_PRESS)*piston_a*y)

    return (d_piston_m_dt, d_piston_t_dt, d_theta_dt, d_dthetadt_dt, d_outlet_m_dt, d_outlet_t_dt), (piston_p, piston_v, piston_rho, s, I_2, T_m, outlet_p, outlet_rho)

def f(t, y, *args):
    return step(t, y, *args)[0]

def period_event(t, y, *args):
    piston_m, piston_t, theta, d_theta_dt, outlet_m, outlet_t = y
    return theta - 2*np.pi

def compute_cycle(t, y, **kwargs):
    piston_in_leak_ratio = kwargs['piston_in_leak_ratio'] if 'piston_in_leak_ratio' in kwargs else 0
    piston_out_leak_ratio = kwargs['piston_out_leak_ratio'] if 'piston_out_leak_ratio' in kwargs else 0

    period_event.terminal = True
    # NOTE: t_eval does not affect solver steps, only output times. Values are interpolated.
    fsol = solve_ivp(f,
                     [t, np.inf],
                     y,
                     args=(piston_in_leak_ratio, piston_out_leak_ratio),
                     method='Radau',
                     max_step=MAX_SOLVER_STEP,
                     events=[period_event])

    assert fsol.success

    return fsol

def compute_equilibrium(
        y_init=(
        TDC_DISTANCE*piston_a*rho_atm,
        ATM_TEMP,
        0,
        2*np.pi * MAINS_F * 2 / MOTOR_POLES,
        REGULATOR_PRESSURE*OUTLET_VOLUME/air.R/ATM_TEMP,
        ATM_TEMP
        ),
        **kwargs
):
    y = list(y_init)
    t = 0
    fsols = []
    y_max_last = None
    y_min_last = None
    while True:
        y[2] = 0  # theta
        fsol = compute_cycle(0, y, **kwargs)
        piston_m, piston_t, theta, d_theta_dt, outlet_m, outlet_t = fsol.y
        params = np.array([piston_m, piston_t, d_theta_dt, outlet_m, outlet_t]).T  # No theta
        y_max = np.max(params, axis=0)
        y_min = np.min(params, axis=0)
        fsols.append(fsol)

        if len(fsols) > 1 and \
                np.all(np.abs((y_max - y_max_last) / y_max) < EQM_TOL) and \
                np.all(np.abs((y_min - y_min_last) / y_min) < EQM_TOL) and \
                np.all(np.abs((params[-1] - params[0]) / params[-1]) < EQM_TOL):
            break

        y_max_last = y_max
        y_min_last = y_min
        t += fsol.t[-1]
        y = fsol.y_events[-1][0]

    return fsols

def concat_sols(sols):
    t_all = [sols[0].t]
    for sol in sols[1:]:
        t_all.append(sol.t + t_all[-1][-1])

    t_all = np.hstack(t_all)
    y_all = list(np.hstack([s.y for s in sols]))

    return t_all, y_all


if __name__ == '__main__':
    piston_in_leak_ratio = 0
    piston_out_leak_ratio = 0

    start_time = perf_counter()
    #fsol = solve_ivp(f, [0, END_TIME], [TDC_DISTANCE*piston_a*rho_atm, ATM_TEMP, 0, 2*np.pi * MAINS_F * 2 / MOTOR_POLES, REGULATOR_PRESSURE*OUTLET_VOLUME/air.R/ATM_TEMP, ATM_TEMP], t_eval=t, method='Radau', max_step=MAX_SOLVER_STEP)
    cProfile.run('''
fsols = compute_equilibrium(y_init=[1.0983972138975649e-05, 350.7055202697288, 6.283185307179587, 185.4405744273075, 0.00011121524980722459, 429.2559995834747],
                            piston_in_leak_ratio=piston_in_leak_ratio,
                            piston_out_leak_ratio=piston_out_leak_ratio)
    ''', sort='cumtime')
    end_time = perf_counter()

    print(f"Done. Took {end_time - start_time}s")
    print(f"Equilibrium y={list(np.array(fsols[-1].y)[:,-1])}")

    t, y = concat_sols(fsols)
    piston_m, piston_t, theta, d_theta_dt, outlet_m, outlet_t = y
    (d_piston_m_dt, d_piston_t_dt, d_theta_dt, d_dthetadt_dt, d_outlet_m_dt, d_outlet_t_dt), (piston_p, piston_v, piston_rho, s, I_2, T_m, outlet_p, outlet_rho)\
        = step(None, y, piston_in_leak_ratio=piston_in_leak_ratio, piston_out_leak_ratio=piston_out_leak_ratio)

    t_sample = np.arange(0, END_TIME, SAMPLE_TIME)
    p_sample = np.interp(t_sample, t, piston_p)
    p_sample = moving_average(p_sample, 4)
    p_sample = np.random.normal(p_sample, 0.03*PSI_TO_PA)
    p_sample = p_sample - p_sample % 0.04*PSI_TO_PA

    plt.title("Piston Pressure")
    plt.xlabel("t (s)")
    plt.ylabel("Pressure (PSI)")
    plt.plot(t, piston_p / PSI_TO_PA, label='Piston')
    plt.plot(t, outlet_p / PSI_TO_PA, label='Outlet')
    plt.legend()
    plt.show()

    plt.title("Mass Flow")
    plt.xlabel("t (s)")
    plt.ylabel("Mass Flow (kg/s)")
    plt.plot(t, d_piston_m_dt, label='Piston')
    plt.plot(t, d_outlet_m_dt, label='Outlet')
    plt.legend()
    plt.show()

    plt.title("Temperature")
    plt.xlabel("t (s)")
    plt.ylabel("T (K)")
    plt.plot(t, piston_t, label='Piston')
    plt.plot(t, outlet_t, label='Outlet')
    plt.legend()
    plt.show()

    plt.title("Slip")
    plt.xlabel("t (s)")
    plt.ylabel("s")
    plt.plot(t, s)
    plt.show()

    plt.title("Flow Rate")
    plt.xlabel("Pressure (Pa)")
    plt.ylabel("dmdt")
    plt.plot(piston_p[-len(fsols[-1].t):], d_piston_m_dt[-len(fsols[-1].t):], label='Piston')
    plt.plot(outlet_p[-len(fsols[-1].t):], d_piston_m_dt[-len(fsols[-1].t):], label='Outlet')
    plt.legend()
    plt.show()

    """
    plt.title("Theta")
    plt.xlabel("Time (s)")
    plt.ylabel("Theta")
    plt.plot(t, theta % (2*np.pi))
    ev = period_event(t, list(np.array(fsols[-1].y)))
    plt.plot(t, ev)
    plt.show()
    """

