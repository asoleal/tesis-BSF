#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deb_base.py
S1.1 — Modelo DEB dinámico de Eriksen (2022) para Hermetia illucens
Extraído de: Eriksen NT (2022) Dynamic modelling of feed assimilation,
growth, lipid accumulation, and CO2 production in black soldier fly larvae.
PLoS ONE 17(10): e0276605.  https://doi.org/10.1371/journal.pone.0276605

Autor: John Leal (asoleal) — tesis doctoral bioconversión BSF
Repositorio: https://github.com/asoleal/tesis-BSF

Uso:
    from deb_base import EriksenParams, EriksenDEB
    p = EriksenParams(N_larvae=700, V_gas_L=3.0, T_celsius=28.0)
    deb = EriksenDEB(p)
    res = deb.simulate_batch_closed(B0=0.014, Ltot0=0.0014, t_max=30.0)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict


@dataclass
class EriksenParams:
    """Parámetros del modelo DEB de Eriksen (2022).
    Unidades base: carbono equivalente [mg C], tiempo [días].
    """
    # --- Tasas metabólicas ---
    a_max: float = 1.4          # día^-1  : tasa específica máxima de asimilación
    m: float = 0.08             # día^-1  : tasa específica de mantenimiento (CO2)
    Y_B: float = 0.44           # adim    : costo de síntesis de biomasa estructural
    Y_L: float = 0.42           # adim    : costo de síntesis de lípidos de almacenamiento

    # --- Logísticas ---
    alpha: float = 0.92         # adim    : coef. reducción asimilación instar 6
    beta: float = 2.02          # adim    : coef. reducción crecimiento estructural

    # --- Biomasa máxima ---
    B_max_0: float = 65.0       # mg C    : biomasa estructural máxima (óptima)
    delta_I: float = 5.0        # adim    : ratio peso entre mudas sucesivas
    t_p_min: float = 13.0       # días    : tiempo mínimo desarrollo neonata→prepupa
    rho: float = 1.0            # mg C/día: tasa decrecimiento B_max tras t_p_min

    # --- Fracciones de conversión (Tabla 1 Eriksen) ---
    delta_C_B: float = 0.49     # fracción C en biomasa estructural orgánica
    delta_C_L: float = 0.79     # fracción C en lípidos (trilaurina)
    delta_L_B: float = 0.10     # fracción lípidos en biomasa estructural (mínimo)
    delta_O: float = 0.90       # fracción orgánica del DW total
    delta_DW_B: float = 0.25    # fracción DW de la biomasa estructural húmeda

    # --- Numérica ---
    dt: float = 0.005           # días    : paso Euler (~7.2 min)

    # --- Batch cerrado (acumulación CO2) ---
    V_gas_L: float = 10.0       # L       : volumen gaseoso efectivo del reactor
    T_celsius: float = 28.0     # °C      : temperatura tropical de operación
    N_larvae: int = 700         # número de larvas


class EriksenDEB:
    """
    Implementación del modelo dinámico DEB de Eriksen (2022) para
    Hermetia illucens.  Resuelve B(t), L_tot(t) y r_CO2(t) vía Euler.
    """

    def __init__(self, p: EriksenParams):
        self.p = p
        self.R_atm = 0.082057     # L·atm·K^-1·mol^-1
        self.M_CO2 = 44.01        # g/mol
        self.M_C = 12.01          # g/mol

        # Pre-calcular moles totales de gas en el batch (ideal)
        T_K = p.T_celsius + 273.15
        self.n_gas_total = (1.0 * p.V_gas_L) / (self.R_atm * T_K)  # mol
        # Factor de conversión: mg C de CO2 → ppm en V_gas_L
        self.ppm_per_mgC_CO2 = (1e-3 / self.M_C) / self.n_gas_total * 1e6

    # ------------------------------------------------------------------
    # Conversión DW (peso seco) ↔ Carbono equivalente
    # ------------------------------------------------------------------
    def B_from_DW(self, X_DW_mg: float) -> float:
        """Dado un peso seco total larva [mg DW], estima B [mg C].
        Asume contenido lipídico mínimo (δ_L,B)."""
        p = self.p
        denom = ((1.0 - p.delta_L_B) / (p.delta_C_B * p.delta_O)
                 + p.delta_L_B / p.delta_C_L)
        return X_DW_mg / denom

    def DW_from_state(self, B_mgC: float, L_tot_mgC: float) -> float:
        """Convierte estado (B, L_tot) en carbono a peso seco total [mg DW]."""
        p = self.p
        return (((1.0 - p.delta_L_B) * B_mgC) / (p.delta_C_B * p.delta_O)
                + L_tot_mgC / p.delta_C_L)

    def WW_from_state(self, B_mgC: float, L_DW_mg: float) -> float:
        """Peso húmedo total [mg WW] — Eq. 22 Eriksen."""
        p = self.p
        B_DW_mg = ((1.0 - p.delta_L_B) * B_mgC) / (p.delta_C_B * p.delta_O)
        return B_DW_mg / p.delta_DW_B + L_DW_mg

    # ------------------------------------------------------------------
    # Determinación de instar y B_max(t)
    # ------------------------------------------------------------------
    def instar(self, B: float, t: float) -> int:
        """Devuelve instar I según Eq. 15 Eriksen."""
        p = self.p
        threshold_moult = (1.0 / p.delta_I) * p.B_max_0
        if B < threshold_moult:
            return 5   # instares 1-5 agrupados (crecimiento exponencial)
        elif B < p.B_max_0:
            return 6
        else:
            return 7   # prepupa (no alimentación)

    def B_max(self, t: float) -> float:
        """Biomasa estructural máxima dependiente del tiempo — Eq. 16."""
        p = self.p
        if t < p.t_p_min:
            return p.B_max_0
        else:
            val = p.B_max_0 - p.rho * (t - p.t_p_min)
            return max(val, 0.0)

    # ------------------------------------------------------------------
    # Tasas instantáneas (mg C · día^-1) para UNA larva
    # ------------------------------------------------------------------
    def rates(self, B: float, L_tot: float, t: float) -> Dict[str, float]:
        """Calcula todas las tasas del modelo para una larva individual."""
        p = self.p
        I = self.instar(B, t)
        Bm = self.B_max(t)

        # --- Asimilación (Eq. 5) ---
        if I < 6:
            a = p.a_max
        elif I == 6:
            ratio = B / Bm if Bm > 0 else 1.0
            a = p.a_max * (1.0 - ratio ** p.alpha) if ratio < 1.0 else 0.0
        else:
            a = 0.0

        r_A = a * B

        # --- Mantenimiento (Eq. 6) ---
        r_CO2_m = p.m * B

        # --- Crecimiento estructural (Eq. 7 + Eq. 10 corregida) ---
        if I <= 6 and Bm > 0:
            mu_B = ((a - p.m) / (1.0 + p.Y_B)) * (1.0 - (B / Bm) ** p.beta)
            mu_B = max(mu_B, 0.0)
        else:
            mu_B = 0.0

        r_B = mu_B * B

        # --- CO2 asociado a crecimiento estructural (Eq. 11) ---
        r_CO2_B = p.Y_B * r_B

        # --- Balance de asimilados para lípidos (Eq. 12) ---
        surplus = r_A - (r_B + r_CO2_m + r_CO2_B)

        if surplus >= 0:
            r_L = surplus / (1.0 + p.Y_L)
            r_CO2_L = p.Y_L * r_L
        else:
            r_L = surplus          # reciclaje negativo
            r_CO2_L = 0.0

        r_CO2_total = r_CO2_m + r_CO2_B + r_CO2_L

        return {
            'I': I,
            'a': a,
            'mu_B': mu_B,
            'r_A': r_A,
            'r_B': r_B,
            'r_L': r_L,
            'r_CO2_m': r_CO2_m,
            'r_CO2_B': r_CO2_B,
            'r_CO2_L': r_CO2_L,
            'r_CO2_total': r_CO2_total,
            'B_max': Bm,
            'surplus': surplus,
        }

    # ------------------------------------------------------------------
    # Integración Euler para UNA larva
    # ------------------------------------------------------------------
    def simulate_individual(self, B0: float, Ltot0: float, t_max: float,
                            t0: float = 0.0) -> Dict[str, np.ndarray]:
        """Simula una larva desde (B0, Ltot0) durante t_max días."""
        p = self.p
        dt = p.dt
        n_steps = int(np.ceil(t_max / dt)) + 1

        t = np.zeros(n_steps)
        B = np.zeros(n_steps)
        L_tot = np.zeros(n_steps)

        t[0] = t0
        B[0] = B0
        L_tot[0] = Ltot0

        rates_keys = ['r_A', 'r_B', 'r_L', 'r_CO2_m', 'r_CO2_B', 'r_CO2_L',
                      'r_CO2_total', 'I', 'a', 'mu_B', 'surplus']
        rates_hist = {k: np.zeros(n_steps) for k in rates_keys}

        for i in range(n_steps - 1):
            rr = self.rates(B[i], L_tot[i], t[i])
            for k in rates_keys:
                rates_hist[k][i] = rr[k]

            # Euler — Eqs. 17-18 Eriksen
            dB = rr['r_B'] * dt
            dLtot = (rr['r_L'] + p.delta_L_B * rr['r_B']) * dt

            B[i+1] = B[i] + dB
            L_tot[i+1] = L_tot[i] + dLtot
            t[i+1] = t[i] + dt

            if B[i+1] < 0:
                B[i+1] = 0.0
            if L_tot[i+1] < 0:
                L_tot[i+1] = 0.0

        rr = self.rates(B[-1], L_tot[-1], t[-1])
        for k in rates_keys:
            rates_hist[k][-1] = rr[k]

        # Conversiones a peso seco / húmedo
        X_DW = np.zeros(n_steps)
        X_WW = np.zeros(n_steps)
        delta_lipid = np.zeros(n_steps)
        delta_DW = np.zeros(n_steps)
        L_DW = np.zeros(n_steps)

        for i in range(n_steps):
            L_DW[i] = L_tot[i] / p.delta_C_L
            X_DW[i] = self.DW_from_state(B[i], L_tot[i])
            X_WW[i] = self.WW_from_state(B[i], L_DW[i])
            delta_lipid[i] = (L_DW[i] / X_DW[i] * 100.0) if X_DW[i] > 0 else 0.0
            delta_DW[i] = (X_DW[i] / X_WW[i] * 100.0) if X_WW[i] > 0 else 0.0

        return {
            't': t, 'B': B, 'L_tot': L_tot,
            'X_DW': X_DW, 'X_WW': X_WW,
            'L_DW': L_DW, 'delta_lipid': delta_lipid, 'delta_DW': delta_DW,
            **rates_hist
        }

    # ------------------------------------------------------------------
    # Batch cerrado: acumulación de CO2 (ppm)
    # ------------------------------------------------------------------
    def simulate_batch_closed(self, B0: float, Ltot0: float, t_max: float,
                              t0: float = 0.0) -> Dict[str, np.ndarray]:
        """
        Simula N larvas idénticas en reactor batch cerrado.
        Devuelve la misma info individual más 'CO2_ppm' acumulado.
        """
        p = self.p
        N = p.N_larvae

        ind = self.simulate_individual(B0, Ltot0, t_max, t0)

        t = ind['t']
        n = len(t)
        CO2_ppm = np.zeros(n)

        # d(ppm) = r_CO2_total_ind * N * (M_CO2/M_C) * ppm_per_mgC_CO2 * dt
        factor = N * (self.M_CO2 / self.M_C) * self.ppm_per_mgC_CO2 * p.dt

        for i in range(n - 1):
            d_ppm = ind['r_CO2_total'][i] * factor
            CO2_ppm[i+1] = CO2_ppm[i] + d_ppm

        batch = ind.copy()
        batch['CO2_ppm'] = CO2_ppm
        batch['r_CO2_total_batch_mgC_d'] = ind['r_CO2_total'] * N
        batch['r_CO2_total_batch_mgCO2_d'] = ind['r_CO2_total'] * N * (self.M_CO2 / self.M_C)
        return batch


if __name__ == "__main__":
    # Ejemplo de ejecución rápida
    params = EriksenParams(
        a_max=1.4, m=0.08, Y_B=0.44, Y_L=0.42,
        alpha=0.92, beta=2.02, B_max_0=65.0,
        delta_I=5.0, t_p_min=13.0, rho=1.0,
        dt=0.005, V_gas_L=3.0, T_celsius=28.0, N_larvae=700,
    )
    deb = EriksenDEB(params)

    # Inicialización desde peso seco conocido
    X_DW0 = 0.03  # mg DW (neonata)
    B0 = deb.B_from_DW(X_DW0)
    Ltot0 = params.delta_L_B * B0

    res = deb.simulate_batch_closed(B0, Ltot0, 30.0)
    print(f"Simulación completa.  Tiempo saturación 5000 ppm:", end=" ")
    idx = np.where(res['CO2_ppm'] >= 5000.0)[0]
    if len(idx):
        print(f"{res['t'][idx[0]]*24*60:.1f} min")
    else:
        print("No alcanzado en 30 días")
