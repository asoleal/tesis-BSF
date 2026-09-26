# patch_readme16.py — agrega §16 al README (reconstruccion del articulo + PINN)
from pathlib import Path
f = Path("README.md")
s = f.read_text()
assert "## 16." not in s, "§16 ya existe"

s16 = """
## 16. Reconstrucción del artículo + capa PINN (2026-09-26)

### Qué se hizo (Fase 3: contenido_art4.tex reconstruido de cero)
El contenido completo previo se había perdido (nunca se commiteó); se reescribió por parches (`patch_*.py`, todos versionados):

- **§2.1** reescrita: DEB + switch de prepupa S5 + módulos (8 ecuaciones, solo entornos core; sin accents por compatibilidad del preamble).
- **§2.2** reescrita: calendario adaptativo de dos pasadas, clip 120–1800 s, tasa en punto medio, ventilación 4 L/min como constraint (S11).
- **§2.3** nueva: capa PINN con 3 roles (corrección de tasa, inferencia de biomasa, CH4) + notas pendientes.
- **§3.3** alineada con §2.2 (quedaba la versión vieja "día a día").
- **§3.4** nueva: validación preliminar contra experimento sep 2025 (figura `validacion_preliminar.pdf` + Cuadro 1).
- **§3.5** nueva: entrenamiento piloto PINN (figura `pinn_entrenamiento.pdf` + ecuación de pérdida).
- **§4 CFD detallada**: 4.1 objetivo/criterio, 4.2 geometría/malla, 4.3 flujo (Stokes penalizado), 4.4 transporte (SUPG), 4.5 métricas, 4.6 verificación numérica, 4.7 escenarios, 4.8 C5.
- **§5** reescrita: dinámica del ciclo + efecto del switch de prepupa + CFD + validación.
- **Conclusiones**: item 1 corregido; items 6 (prepupa), 7 (validación), 8 (PINN) agregados.
- Fix babel: `\\,\\%` → `\\%` (2 reemplazos).
- Resultado: `main_art4.pdf` compila sin errores (12+ páginas, 17 refs).

### Resultados principales
| Resultado | Valor |
|---|---|
| Picos CO2 cierres (tope NDIR 5000) | E2: 4978 ppm, E5: 4604 ppm |
| Duraciones de cierre | 30 min (inicio) → 2.5–7 min (pico) → alargamiento en ayuno |
| Emisiones ciclo | E2 (N=400): 0.59 mol CO2; E5 (N=700): 1.04 mol |
| Switch prepupa S5 | L final 209→54 mg/larva; emisiones E2 1.37→0.59 mol |
| Validación (modelo vs obs neto) | 210/289/129/15 vs 89–309 / 124–165 / 22–103 / 0–14 ppm/min |
| PINN piloto (red 4-8-1, 49 par) | J 4.48→17.29; δ = 0.53/0.48/0.24/0.10 (días 9/11/13/17) |
| PINN: tasas corregidas | 112/140/31/1.5 ppm/min — dentro del rango observado los 4 días |
| PINN: CO2 ciclo | 1.05 → 1.00 mol (tope de carbono 2×) |

### Qué queda pendiente
1. **PINN**: reemplazar datos sintéticos por digitalización de Eriksen 2022/2024 y reentrenar; LOO diferido (49 params vs ≤6 puntos/pliegue); calibrar λ por CV.
2. **S8**: recalibrar módulo microbiano con curvas de control completas (descomposición larvaria/microbiana no identificable aún).
3. **S6**: biomasa pesada en experimentos finales → verificar B(t) y parámetros DEB.
4. **S11/S1**: calibrar V_air por trazador; CH4 contra referencia.
5. **Artículo (parches listos, no aplicados)**: Resumen (añadir validación + PINN); cosméticos (Fig 2 `width=0.92\\textwidth`, mover caveat de Fig 3b al caption).
6. Cosmético: aviso `float too large` Figura 2 (no rompe, ocupa página entera).

### Archivos nuevos (todos commiteados)
- `simulacion/pinn_entrenamiento.py` — entrenamiento piloto (semilla 7, deterministico).
- `imagenes/pinn_entrenamiento.pdf`, `resultados/pinn_resumen.md`.
- `patch_s21.py` … `patch_s35_pinn.py` — parches aplicados al tex.
"""

f.write_text(s + s16)
print("OK: README §16 agregada")
