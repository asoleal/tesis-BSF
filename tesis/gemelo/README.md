# Gemelo digital — Bioconversión de residuos con Hermetia illucens

Modelado y predicción de emisiones de GEI (CO2, CH4) en batch controlado
con monitoreo de temperatura, humedad y ventilación. Sensores: Winsen
MH-410D (CO2, 0-5000 ppm) y MH-441D (CH4, 0-5 %vol). Camara: panera 11.9 L.

## Estructura

- `modelo_2d/` — seccion transversal (py-pde): gradientes, ubicacion de
  sensores, ventilacion. `co2_cama_2d.py`, `co2_ch4_ventilacion.py`
- `modelo_0d/` — nucleo del gemelo (scipy): `gemelo_0d.py` (batch 14 dias),
  `calibrar_camara.py` (flujos desde datos reales), `gemelo_camara.py`
  (validacion R2=0.81 + tabla de diseno ACH)
- `datos/` — `flujos_reales.csv` (calibracion), `sintetico_base.csv`
- `resultados/` — figuras generadas (alimentan articulos/*/figuras/)

## Hallazgos clave

- Sin ventilacion la cama supera 8 % de CO2 (2D); ventilar la cabecera
  sola no baja de ~3.5 %: la difusion en la cama es el cuello de botella
- Flujos reales camara cerrada: 58-262 mL CO2/h por 750 larvas
- Con larvas D4 activas se requieren >= 8 ACH para no saturar el MH-410D
- Batch 17 D4: flujo neto negativo -> anomalia marcada (revisar)
- MH-441D (NDIR hidrocarburos) responde a COVs de fermentacion: CH4
  requiere validacion (test etanol, trampa CO2, idealmente GC)

## Hoja de ruta

1. [x] Difusion CO2 2D sin ventilacion
2. [x] Ventilacion forzada + CH4 en 2D (curva de diseno)
3. [x] Modelo 0D del batch completo (7 estados acoplados)
4. [x] Calibracion con datos reales (camara cerrada) + validacion
5. [ ] Validacion de CH4 (protocolo de interferencias)
6. [ ] Gemelo completo calibrado en modo continuo (batch de 14 dias)
7. [ ] Dashboard de visualizacion del gemelo
