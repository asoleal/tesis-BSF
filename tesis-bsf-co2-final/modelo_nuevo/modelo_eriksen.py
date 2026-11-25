#!/usr/bin/env python3
"""
modelo_eriksen.py — Nucleo DEB de Eriksen (puro), 2 estados: [B, Ltot].

Unidades: mg C por larva, d^-1.
tau = t + 4.5: edad fisiologica desde neonato.
Prepupa desactivada (f_mant=1.0, k_cat=0.0).
"""

import numpy as np


def Bmax_t(tau: float, p: dict) -> float:
    """Techo de biomasa a edad fisiologica tau."""
    return max(p["Bmax0"] - p["rho"] * max(tau - p["tp_min"], 0.0), 1e-12)


def instar(B: float, tau: float, p: dict) -> int:
    """Instar operacional: 5 (pequena), 6 (crecimiento), 7 (techo)."""
    if B < p["Bmax0"] / p["delta_I"]:
        return 5
    return 7 if B >= Bmax_t(tau, p) else 6


def tasas_larva(t: float, B: float, Ltot: float, p: dict, f_dieta: float = 1.0):
    """
    Devuelve tasas del nucleo Eriksen para UNA larva.
    f_dieta: factor de calidad del alimento (1.0 = D1, <1 = D4).
    """
    tau = t + 4.5
    B = max(float(B), 0.0)
    bm = Bmax_t(tau, p)
    I = instar(B, tau, p)

    # Tasa de asimilacion
    if I < 6:
        a = p["amax"]
    elif I == 6:
        a = p["amax"] * max(1.0 - (B / bm) ** p["alpha"], 0.0)
    else:
        a = 0.0

    a_eff = a * f_dieta
    rA = a_eff * B
    G = max(1.0 - (B / bm) ** p["beta"], 0.0)
    muB = ((a_eff - p["m"]) / (1.0 + p["YB"])) * G

    if I <= 6:
        rB = muB * B
        rCB = p["YB"] * rB
        rCm = p["m"] * B
        sL = rA - (rB + rCm + rCB)
        rcat = 0.0
    else:
        # Instar 7: sin asimilacion, mantenimiento pleno, sin catabolismo estructural
        rB = -p["k_cat_prepupa"] * B
        rCB = 0.0
        rCm = p["m"] * p["f_mant_prepupa"] * B
        sL = rA - rCm
        rcat = p["k_cat_prepupa"] * B

    rL = sL / (1.0 + p["YL"]) if sL >= 0.0 else sL
    rCL = p["YL"] * max(rL, 0.0)
    rCO2_C = rCm + rCB + rCL + rcat

    dB = rB
    dLtot = rL + p["delta_LB"] * rB

    return {
        "instar": I, "a_eff": a_eff, "rA": rA, "rCm": rCm,
        "rB": rB, "rCB": rCB, "sL": sL, "rL": rL, "rCL": rCL,
        "rCO2_C": rCO2_C, "dB": dB, "dLtot": dLtot,
    }


def rhs_eriksen(t: float, y: np.ndarray, p: dict, f_dieta: float = 1.0) -> np.ndarray:
    """RHS del sistema Eriksen puro: [dB/dt, dLtot/dt]."""
    B, Ltot = y[0], y[1]
    larva = tasas_larva(t, B, Ltot, p, f_dieta)
    return np.array([larva["dB"], larva["dLtot"]])


def estado_inicial_desde_dw(dw_mg: float, p: dict) -> np.ndarray:
    """Convierte peso seco inicial (mg) a estados [B0, Ltot0]."""
    B0 = dw_mg * p["delta_CB"] * p["delta_O"]
    Ltot0 = B0 * p["delta_LB"] * (p["delta_CL"] / (p["delta_CB"] * p["delta_O"]))
    return np.array([B0, Ltot0])


def B_Ltot_a_PH(B: float, Ltot: float, p: dict, f_MS: float = 0.325) -> float:
    """Convierte estados del modelo a peso humedo (PH) en mg."""
    return ((1.0 - p["delta_LB"]) * B / (p["delta_CB"] * p["delta_O"]) + Ltot / p["delta_CL"]) / f_MS
