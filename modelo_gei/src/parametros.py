"""Parametros del modelo de bioconversion con H. illucens.
Seccion 01 - modelo_gei. Unidades: dias, mg (biomasa), g (gases), C-eq.
'calibrar=True' -> se estima con nuestros datos del batch.
"""

NUCLEO_DEB = {
    "a_max":   dict(valor=1.30, rango=(1.1, 1.5), unidad="d^-1",
                    fuente="eriksen2022", calibrar=True,
                    nota="buenas condiciones; 0.8-1.1 estres hidrico; 0.25-1.21 limitacion"),
    "Y_B":     dict(valor=0.44, rango=(0.35, 0.44), unidad="-",
                    fuente="laganaro2021", calibrar=True,
                    nota="costo de crecimiento biomasa estructural"),
    "m":       dict(valor=0.08, rango=(0.08, 0.13), unidad="d^-1",
                    fuente="laganaro2021", calibrar=True,
                    nota="mantenimiento (CO2); sube con sustrato pobre"),
    "Y_L":     dict(valor=0.42, rango=None, unidad="-",
                    fuente="eriksen2022", calibrar=False,
                    nota="costo de sintesis de lipidos"),
    "B_max0":  dict(valor=65.0, rango=(65.0, 90.0), unidad="mg",
                    fuente="eriksen2022", calibrar=True,
                    nota="peso max biomasa estructural, condiciones optimas"),
    "t_p_min": dict(valor=13.0, rango=None, unidad="d",
                    fuente="eriksen2022", calibrar=False,
                    nota="tiempo minimo a prepupa"),
    "rho":     dict(valor=1.0, rango=None, unidad="mg/d",
                    fuente="eriksen2022", calibrar=False,
                    nota="caida de B_max despues de t_p_min"),
    "delta_I": dict(valor=5.0, rango=(4.0, 5.0), unidad="-",
                    fuente="eriksen2022", calibrar=False,
                    nota="razon de peso entre instares"),
    "alpha":   dict(valor=1.0, rango=(0.28, 3.3), unidad="-",
                    fuente="eriksen2022", calibrar=True,
                    nota="downregula asimilacion en instar 6"),
    "beta":    dict(valor=1.0, rango=(1.0, 2.0), unidad="-",
                    fuente="eriksen2022", calibrar=True,
                    nota="downregula crecimiento -> reparto a lipidos"),
}

ESTEQUIOMETRIA = {
    "delta_C_B":   0.49,   # C en biomasa estructural
    "delta_C_L":   0.74,   # C en lipidos
    "delta_C_CO2": 0.27,   # C en CO2
    "delta_L_B":   0.10,   # lipidos estructurales
    "delta_O":     0.90,   # fraccion organica (SV)
    "delta_DW_B":  0.25,   # MS de la biomasa estructural
}

RANGOS_OPERACION = {
    "T_sustrato":       dict(optimo=(25.0, 30.0), viable=(25.0, 35.0), unidad="degC",
                             fuente="shumo2019"),
    "humedad_sustrato": dict(optimo=0.75, viable=(0.45, 0.85), unidad="-",
                             fuente="chen2019; bekker2021"),
    "HR_aire":          dict(optimo=(0.60, 0.75), viable=None, unidad="-",
                             fuente="literatura de crianza"),
    "pH_sustrato":      dict(optimo=(6.0, 8.0), viable=(2.0, 11.0), unidad="-",
                             fuente="pang2020"),
    "racion":           dict(optimo=100.0, viable=(12.5, 200.0), unidad="mg/larva/d",
                             fuente="diener2009"),
}

# Pendientes de calibrar (aportes propios):
#   Q10 (respiracion BSFL vs T) - no existe en literatura
#   k_CH4 y f(T), f(W), f(pH) del modulo de gases
