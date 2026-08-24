# Modelo y simulación — GEI en bioconversión con H. illucens

Simulación del modelo de Eriksen (DEB) + módulo GEI, análisis de
identificabilidad in-silico (EKF / PINN-UDE) y diseño experimental.

## Estructura
- `docs/` — secciones del modelo (markdown, una por paso)
- `src/` — código Python
- `resultados/figuras/` — salidas

## Hoja de ruta (estado)
- [ ] 01 — Parámetros de literatura (docs/01_parametros.md + src/parametros.py)
- [ ] 02 — Modelo Eriksen: ecuaciones + EDO (docs/02_modelo_eriksen.md + src/eriksen.py)
- [ ] 03 — Datos sintéticos + ruido (src/sintetico.py)
- [ ] 04 — Identificabilidad: EKF y PINN/UDE (docs/03_identificabilidad.md)
- [ ] 05 — Plan experimental derivado (docs/04_plan_experimental.md)

Regla: cada sección se aprueba antes de seguir (ok → siguiente).
