# Gemelo digital — Bioconversión de residuos con Hermetia illucens

Modelado y predicción de emisiones de GEI (CO2, CH4) en un batch
controlado con monitoreo de temperatura, humedad y ventilación.

## Estructura

- `modelo_2d/` — sección transversal de la caja (py-pde): gradientes de
  CO2/CH4, ubicación de sensores, ventilación. Verificado: `co2_cama_2d.py`
- `modelo_0d/` — núcleo del gemelo en tiempo real (balances masa/energía,
  scipy). Se calibra con los datos de sensores. [pendiente]
- `datos/` — datos crudos de sensores (cuando existan)
- `resultados/` — figuras y CSV generados; alimentan `articulos/*/figuras/`

## Uso

    python -m venv .venv && source .venv/bin/activate
    pip install py-pde numpy matplotlib scipy
    python modelo_2d/co2_cama_2d.py

## Hoja de ruta

1. [x] Difusión CO2 2D sin ventilación (justifica ventilación forzada)
2. [ ] Ventilación forzada + CH4 en el modelo 2D
3. [ ] Modelo 0D (gemelo) con fuentes metabólicas ligadas a crecimiento larvario
4. [ ] Calibración con datos reales + validación (RMSE, IC)
5. [ ] Dashboard de visualización del gemelo
