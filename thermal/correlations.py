import numpy as np
import scipy as sp


def CircPipeFF(Re, D_h, rough):
    #return circular pipe friction factor (turbulent: Zigrang and Sylvester)
    if (Re < 2300):
        return 64 / Re
    else:
        return (-2 * np.log10(2 * rough / (7.54 * D_h) - 5.02 / Re * np.log10(2*rough / (7.54 * D_h) + 13 / Re))) ** (-2)

def RectPipeFF(Re, H, W, rough):
    #return recatngular pipe friction factor from Nellis and Klein, eq 5-60 and 5-65
    AR = H / W
    D_h = 2 * H * W / (H + W)
    if (Re < 2300):
        return 96 / Re * (1 - 1.3553 * AR + 1.9467 * AR ** 2 - 1.7012 * AR ** 3 + 0.9564 * AR ** 4 - 0.2537 * AR ** 5)
    else:
        return (-2 * np.log10(2 * rough / (7.54 * D_h) - 5.02 / Re * np.log10(2 * rough / (7.54 * D_h) + 13 / Re))) ** (-2)

def CircPipeNu(Re, ff, Pr):
    #return circular pipe Nusselt number from Nellis and Klein eq 5-76 and Taler 2018
    if (Re < 2300):
        return 4.36
    else:
        return 5.31 + 0.221 * (Re * Pr) ** 0.8174

def RectPipeNu(Re, H, W, ff, Pr):
    #return rectangular pipe Nusselt number
    AR = H / W
    if (Re < 3000):
        return 8.235 * (1 - 2.042 * AR + 3.085 * AR ** 2 -2.477 * AR ** 3 + 1.058 * AR ** 4 - 0.186 * AR ** 5)
    else:
        return 5.31 + 0.221 * (Re * Pr) ** 0.8174

def RprimeCondCyl(D_in, D_out, k_wall):
    #def cylindrical wall conduction thermal resistance [K/Wm]
    return np.log(D_out / D_in) / (2 * np.pi * k_wall)

def RCondWall(L, A_c, k_wall):
    #define planar conduction thermal resistance [K/W]
    return L / (k_wall * A_c)

def RprimeConvCyl(Nu, k_fluid, D):
    #define convection thermal resistance for a fluid in a pipe [K/Wm]
    h_bar = Nu * k_fluid / D
    per = np.pi * D
    return 1 / (h_bar * per)

def RConvWall(Nu, L, A_c, k_fluid):
    #def convection thermal resistance on a planar wall [K/W]
    #L is the characteristic length, hydraulic diameter for internal flows
    h_bar = Nu * k_fluid / L
    return 1 / (h_bar * A_c)

def Rrad(T_sur, T_s, A_s, VF, emis):
    #def exact radiation thermal resistance [K/W]
    return 1 / (A_s * emis * (T_s ** 2 + T_sur ** 2) * (T_sur + T_s) * sp.constants.sigma)

def Rrad_2surfaces(T_1, T_2, A_1, A_2, emis_1, emis_2, VF_12):
    #def radiation thermal resistance between 2 grey bodies [K/W]
    R_rad = ((1 - emis_1) / (A_1 * emis_1) + 1 / (A_1 * VF_12) + (1 - emis_2) /
             (A_2 * emis_2)) / ((T_1 ** 2 + T_2 ** 2) * (T_1 + T_2) * sp.constants.sigma)
    return R_rad
