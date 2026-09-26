# Supuestos operativos (validacion preliminar, articulo 4)

Estado: PRELIMINAR (2026-09-26). Cada supuesto nombra su parametro en
simulacion/bioconversion_ode.py; el experimento final solo cambia valores.

## Sistema / condiciones del experimento
- S1  V_air = 12.1 L. Panera sellada: base 30x19 cm, tapa 36x25 cm, alto 17 cm
      (~12.4 L geometricos menos ~0.25 L de lecho). PENDIENTE: medir por llenado de agua.
      Parametro: FIS['Vair']. (Camara formal del articulo: 2.5 L.)
- S2  N = 700 larvas, constante (mortalidad despreciable en 17 d). Parametro: N0.
- S3  Alimento 250 g por reposicion, AD LIBITUM (se repone en cada medicion).
      Parametro: S0 (DM0/W0 = 0.4/0.6).
- S4  T ~ 27 C de laboratorio, sin control termico en el preliminar. FIS['Ta'], FIS['Tref'].

## Biologia
- S5  Prepupa: asimilacion cesa con logistica centrada en dia 12 (ancho 1 d);
      mantenimiento cae a x0.13 -> 0.64 mg/larva/d, anclado a la medicion de ayuno
      (dia 17: 14 ppm/min). Parametro: PREP = dict(t_p=12.0, ancho=1.0, m_suelo=0.13).
- S6  DEB de Eriksen et al. (2022, 2024). Sin biomasa larval medida: B(t) es prediccion
      no verificada. Parametro: PHY.

## Medicion / datos
- S7  Tasa neta larval = tratamiento - control del MISMO dia (mismo alimento, misma edad).
- S8  Modulo microbiano recalibrado contra las 8 curvas de control (kref, kmax, YCO2).
- S9  CO2: NDIR con tope 6000 ppm (el plateau del CSV es del instrumento, no del proceso).
- S10 CH4: TGS2611 no cuantitativo -> indicador solamente; NO entra al ajuste ni al PINN
      hasta calibrar (Mitchell 2024; Kiplimo 2024).
- S11 Ventilacion entre cierres: 4 L/min (ambas bombas de 2 L/min). Con 1 L/min la base estacionaria (c_in + R/Q) supera 4000 ppm en el pico del ciclo y ningun cierre puede mantenerse bajo 5000 ppm aun con Δ en el piso; verificado con el gemelo (E5: pico 4889 ppm en sandbox; 4604 en local).

- S12 Dietas: D1 = alimento comercial pollo de engorde (~20% PC en MS),
  referencia en eriksenDynamicModellingFeed2022/2024; D4 = mezcla de
  cascaras de naranja y platano, pulpas y frutas de descarte fermentada
  con Bacillus. Proporciones D4 40/30/30 peso humedo = SUPUESTAS, ajus-
  tar al registro real de la campana sep 2025 antes de envio.
