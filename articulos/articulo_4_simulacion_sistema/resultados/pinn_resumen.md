# Resumen entrenamiento piloto PINN (FICTICIO)

Generado: 2026-09-25 19:33

- Red: 4->8->1 tanh, 49 parametros, numpy + L-BFGS-B
- Pre-train: 48 puntos SINTETICOS tipo literatura (semilla 7). REEMPLAZAR por digitalizacion Eriksen 2022/2024
- Fine-tune: 7 tasas propias (septiembre 2025)
- J pre-train = 4.48 | J fine-tune = 17.29
- delta_medio = 0.37 (rango 0.10-0.53)
- CO2 ciclo: baseline 1.05 mol, corregido 1.00 mol (tope 2.0x baseline)
- LOO: NO aplicable (parametros > datos por pliegue); diferido

## Notas pendientes
1. Reemplazar datos sinteticos por datos digitalizados de literatura.
2. Reentrenar con curvas de control completas (S8) al llegar experimentos finales.
3. Fijar LAM_B, LAM_BG y LAM_C por validacion cruzada.
4. Con n pequeno el ajuste es pobre por construccion: el valor esta en la arquitectura de perdida.
5. El componente microbiano aislado no es identificable (r_mic ~ 0 en la replica); se corrige la tasa total.
