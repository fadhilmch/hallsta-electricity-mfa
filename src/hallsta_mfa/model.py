"""Energy and exergy model of electricity use at Holmen Hallsta paper mill (2023).

All flows are in GWh/y. `run(params)` is a pure function: give it a dict of inputs
(single numbers or NumPy arrays) and it returns every flow of the balance.
"""
import numpy as np
import CoolProp.CoolProp as CP

# Base-case inputs (see notebook section 1 for where each number comes from)
BASE = dict(
    E_tot=1450.0, paper_t=444494.0, pulp_t=421883.0, DH_export=11.877,
    V_wwtp=18300*365, V_ff=7288861-18300*365,
    sh_TMP=0.70, sh_PM=0.15, sh_EB=0.10, sh_OTH=0.05,
    P_wood=2.5, P_bleach=3.0, P_wwtp=2.5, aux_TMP=150.0,
    m_rec=100.0, y_EB=1.5, m_PM=95.0, m_wood=1.5, PM11_frac=14/22,
    h_prod=7200.0, h_wwtp=8760.0, f_dist=0.01, f_steamloss=0.04,
    p_steam=3.4e5, T_cond=90.0, T_TMPwaste=80.0, T_PMwaste=55.0, T_HW=60.0,
    T_DHs=85.0, T_DHr=45.0, T_eff=35.0, T_ff=30.0, T0=7.0, T_ref=7.0, cpv=4.17)

def f_sens(Th, T0):
    # exergy factor of sensible heat released from Th down to T0 (°C)
    Th, T0 = np.asarray(Th, float) + 273.15, np.asarray(T0, float) + 273.15
    with np.errstate(divide='ignore', invalid='ignore'):
        f = 1 - T0*np.log(Th/T0)/(Th - T0)
    return np.where(Th > T0 + 1e-6, f, 0.0)

def steam_v(p, Tc, T0):
    # vectorised steam properties (rounded like the original model)
    p = np.round(np.asarray(p, float), -2); Tc = np.round(np.asarray(Tc, float), 1)
    hg = CP.PropsSI('H','P',p,'Q',1,'Water'); sg = CP.PropsSI('S','P',p,'Q',1,'Water')
    hf = CP.PropsSI('H','T',Tc+273.15,'P',p,'Water'); sf = CP.PropsSI('S','T',Tc+273.15,'P',p,'Water')
    d = hg - hf
    return d/3.6e6, 1 - (np.asarray(T0)+273.15)*(sg-sf)/d

def run(p):
    r = {}; E = p['E_tot']
    tot = p['sh_TMP'] + p['sh_PM'] + p['sh_EB'] + p['sh_OTH']               # rescale shares to 100 %
    TMP, PMtot, EB, OTH = (E*p[k]/tot for k in ['sh_TMP','sh_PM','sh_EB','sh_OTH'])
    # ---- layer 1: electricity end uses
    bleach = p['P_bleach']*p['h_prod']/1000; PM = PMtot - bleach
    wood = p['P_wood']*p['h_prod']/1000; wwtp = p['P_wwtp']*p['h_wwtp']/1000; dist = p['f_dist']*E
    k = np.minimum(1.0, OTH/(wood + wwtp + dist))                           # named items must fit in bucket
    wood, wwtp, dist = wood*k, wwtp*k, dist*k
    util = OTH - wood - wwtp - dist                                         # GAP 1 (residual)
    aux = p['aux_TMP']*p['pulp_t']/1e6
    r.update(el_TMP=TMP, el_TMP_aux=aux, el_TMP_ref=TMP-aux, el_PM=PM, el_bleach=bleach, el_EB=EB,
             el_wood=wood, el_wwtp=wwtp, el_util=util, el_dist=dist,
             el_PM11line=PMtot*p['PM11_frac'], el_PM12line=PMtot*(1-p['PM11_frac']))
    # ---- layer 2: steam loop
    dh, fx = steam_v(p['p_steam'], p['T_cond'], p['T0'])
    st_rec = p['m_rec']*p['h_prod']*dh/1000
    eff_EB = np.minimum(p['y_EB']*dh, 0.995)
    st_EB = EB*eff_EB; supply = st_rec + st_EB
    st_PM = p['m_PM']*p['h_prod']*dh/1000; st_wood = p['m_wood']*p['h_prod']*dh/1000
    st_loss = p['f_steamloss']*supply
    st_HW = supply - st_PM - st_wood - st_loss                              # GAP 2 (residual)
    r.update(dh=dh, fx_steam=fx, eff_EB=eff_EB, st_rec=st_rec, st_EB=st_EB, loss_EB=EB-st_EB,
             st_supply=supply, st_PM=st_PM, st_wood=st_wood, st_loss=st_loss, st_HW=st_HW,
             TMP_waste=TMP-st_rec, PM_waste=PM+st_PM, rec_ratio=st_rec/supply, TMP_rec_eff=st_rec/TMP,
             SEC_mill=E/p['paper_t']*1e6, SEC_TMP=TMP/p['pulp_t']*1e6, SEC_PM=PMtot/p['paper_t']*1e6)
    # ---- layer 3: final sinks
    water = (p['V_wwtp']*p['cpv']*(p['T_eff']-p['T_ref']) + p['V_ff']*p['cpv']*(p['T_ff']-p['T_ref']))/3.6e6
    r.update(sink_water=water, sink_DH=p['DH_export'], sink_air=E - water - p['DH_export'])   # GAP 3 (residual)
    # ---- exergy (section 6)
    T0 = p['T0']
    fTMP, fPM, fHW = f_sens(p['T_TMPwaste'], T0), f_sens(p['T_PMwaste'], T0), f_sens(p['T_HW'], T0)
    Tlm = (p['T_DHs']-p['T_DHr'])/np.log((p['T_DHs']+273.15)/(p['T_DHr']+273.15))
    fDH = 1 - (T0+273.15)/Tlm
    X = {'TMP plant': (TMP, st_rec*fx, (TMP-st_rec)*fTMP),
         'Paper machines': (PM + st_PM*fx, 0*E, (PM+st_PM)*fPM),
         'Electric boilers': (EB, st_EB*fx, 0*E),
         'Steam network': (supply*fx, (st_PM+st_wood+st_HW)*fx, 0*E),
         'Hot water & heating': (st_HW*fx, st_HW*fHW, 0*E),
         'Bleaching': (bleach, 0*E, 0*E), 'WWTP': (wwtp, 0*E, 0*E),
         'Wood handling': (wood + st_wood*fx, 0*E, 0*E),
         'Utilities & other': (util, 0*E, 0*E), 'Internal grid losses': (dist, 0*E, 0*E)}
    for name, (xi, xu, xw) in X.items():
        r[f'Xin|{name}'], r[f'Xuse|{name}'], r[f'Xwaste|{name}'] = xi, xu, xw
        r[f'Xdest|{name}'] = xi - xu - xw
        psi = np.where(np.asarray(xi) > 0, xu/np.where(np.asarray(xi) > 0, xi, 1), 0)
        r[f'psi|{name}'] = psi; r[f'IP|{name}'] = (1-psi)*(xi-xu)
    r['Xdest_total'] = sum(r[f'Xdest|{n}'] for n in X)
    r['Xwaste_total'] = sum(r[f'Xwaste|{n}'] for n in X)
    r['X_eff'] = (p['V_wwtp']*p['cpv']*np.maximum(p['T_eff']-T0,0)*f_sens(p['T_eff'],T0) +
                  p['V_ff']*p['cpv']*np.maximum(p['T_ff']-T0,0)*f_sens(p['T_ff'],T0))/3.6e6
    r['X_DH'] = p['DH_export']*fDH
    r.update(f_TMPw=fTMP, f_PMw=fPM, f_HW=fHW, f_DH=fDH, f_eff=f_sens(p['T_eff'], T0))
    # ---- levers
    r['save_rec90'] = np.maximum(0, 0.90*supply - st_rec)/eff_EB
    r['gap_SEC'] = np.maximum(0, r['SEC_TMP'] - 2150)*p['pulp_t']/1e6
    r['gap_SEC_net'] = r['gap_SEC']*(1 - r['TMP_rec_eff']/eff_EB)
    return {k: (float(v) if np.ndim(v) == 0 else v) for k, v in r.items()}

PROCS = ['TMP plant','Paper machines','Electric boilers','Hot water & heating','Steam network',
         'Bleaching','WWTP','Wood handling','Utilities & other','Internal grid losses']

# Uncertainty ranges used by the Monte Carlo step (notebook section 7a).
# name: (distribution, a, b, c)
#   norm: mean, sd | tri: low, mode, high | uni: low, high
DIST = {
 'E_tot': ('norm', 1450, 14.5, None), 'pulp_t': ('norm', 421883, 4218.83, None), 'paper_t': ('norm', 444494, 4444.94, None),
 'DH_export': ('norm', 11.877, 0.59385, None), 'V_wwtp': ('norm', BASE['V_wwtp'], 0.05*BASE['V_wwtp'], None),
 'V_ff': ('norm', BASE['V_ff'], 0.10*BASE['V_ff'], None),
 'sh_TMP': ('tri', 0.65, 0.70, 0.75), 'sh_PM': ('tri', 0.12, 0.15, 0.22), 'sh_EB': ('tri', 0.05, 0.10, 0.12), 'sh_OTH': ('tri', 0.02, 0.05, 0.08),
 'P_wood': ('tri', 2.0, 2.5, 3.0), 'P_bleach': ('tri', 2.4, 3.0, 3.6), 'P_wwtp': ('tri', 2.0, 2.5, 3.0), 'aux_TMP': ('tri', 120, 150, 180),
 'm_rec': ('tri', 85, 100, 115), 'y_EB': ('tri', 1.45, 1.50, 1.53), 'm_PM': ('uni', 80, 110, None), 'm_wood': ('tri', 1.2, 1.5, 1.8),
 'PM11_frac': ('uni', 0.525, 0.66, None), 'h_prod': ('tri', 6800, 7200, 7600), 'f_dist': ('uni', 0.005, 0.02, None),
 'f_steamloss': ('uni', 0.02, 0.06, None), 'p_steam': ('uni', 3.3e5, 3.6e5, None), 'T_cond': ('uni', 80, 100, None),
 'T_TMPwaste': ('uni', 70, 100, None), 'T_PMwaste': ('uni', 45, 65, None), 'T_HW': ('uni', 50, 70, None),
 'T_eff': ('uni', 28, 40, None), 'T_ff': ('uni', 20, 40, None)}
