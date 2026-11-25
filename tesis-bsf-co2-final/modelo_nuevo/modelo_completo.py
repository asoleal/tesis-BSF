#!/usr/bin/env python3
"""
modelo_completo.py — DEB (Eriksen) + lecho microbiano (Monod) + pastoreo.
6 estados: [B, Ltot, D_C, X, C_CO2, C_residuo].
Unidades: B/Ltot en mg C/larva; D_C, X, C_CO2, C_residuo en g C/camara.
"""

import numpy as np
from modelo_eriksen import tasas_larva, estado_inicial_desde_dw, B_Ltot_a_PH


def tasas_lecho(D_C: float, X: float, T_C: float, p_lecho: dict):
    """Cinetica heterotrofica: crecimiento Monod + decaimiento endogeno."""
    D_C = max(float(D_C), 0.0)
    X = max(float(X), 0.0)
    phi_D = D_C / (p_lecho["K_D"] + D_C) if D_C > 0.0 else 0.0
    factor_T = p_lecho["Q10"] ** ((T_C - p_lecho["T_ref_C"]) / 10.0)
    factor_theta = p_lecho["theta"] / (p_lecho["K_theta"] + p_lecho["theta"])
    k_eff = p_lecho["k_ref"] * factor_T * factor_theta
    rdeg = k_eff * X * D_C / (p_lecho["K_SM"] + D_C) if D_C > 0.0 else 0.0
    r_endog = p_lecho["k_d"] * X
    rCmic = p_lecho["Y_Cmic"] * rdeg + r_endog
    return {
        "phi_D": phi_D,
        "k_eff": k_eff,
        "rdeg": rdeg,
        "rCmic": rCmic,
        "dD_C": -rdeg,
        "dX": p_lecho["Y_X"] * rdeg - r_endog,
    }


def rhs_completo(t: float, y: np.ndarray,
                 p_larva: dict, p_lecho: dict, p_montaje: dict,
                 dieta: str, N: int) -> np.ndarray:
    """
    RHS del sistema completo.
    y = [B, Ltot, D_C, X, C_CO2, C_residuo]
    """
    B, Ltot, D_C, X = y[0], y[1], y[2], y[3]
    T_C = p_montaje["T_C"]
    f_dieta = p_lecho.get("f_dieta", 1.0)
    gamma = p_lecho.get("gamma", 0.0)

    # Nucleo larvario
    larva = tasas_larva(t, B, Ltot, p_larva, f_dieta)

    # Lecho microbiano
    micro = tasas_lecho(D_C, X, T_C, p_lecho)

    # Pastoreo bacteriano
    pastoreo = 0.0
    if gamma > 0 and N > 0:
        rA_total = N * larva["rA"] / 1000.0  # g C camara-1 d-1
        if rA_total > 0 and X > 0:
            pastoreo = gamma * rA_total * X / (p_lecho["K_G"] + X)

    # Derivadas
    dB = larva["dB"]
    dLtot = larva["dLtot"]
    dD_C = -(N * larva["rA"] / 1000.0) - micro["rdeg"]
    dX = micro["dX"] - pastoreo
    dCO2 = N * larva["rCO2_C"] / 1000.0 + micro["rCmic"]
    dresiduo = (1.0 - p_lecho["Y_X"] - p_lecho["Y_Cmic"]) * micro["rdeg"] + pastoreo

    return np.array([dB, dLtot, dD_C, dX, dCO2, dresiduo])


def carbono_a_ppm_min(R_C_g_d: np.ndarray, V_aire_L: float, T_K: float,
                      P_Pa: float = 101325.0) -> np.ndarray:
    """Convierte g C d^-1 a pendiente CO2 en ppm min^-1."""
    R_u = 8.314462618
    mol_d = np.asarray(R_C_g_d) / 12.0
    return mol_d / (V_aire_L / 1000.0) * (R_u * T_K / P_Pa) * 1e6 / 1440.0
