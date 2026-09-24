# Artículo 4 — Simulación del sistema de bioconversión instrumentada

## 1. Qué es esto

Dos modelos complementarios del sistema de bioconversión de residuos con
*Hermetia illucens* (cámara instrumentada, V_aire = 2.5 L, geometría
0.26 × 0.19 × 0.18 m, flujo de ventilación nominal Q = 1 L/min):

| Modelo | Archivo | Qué responde |
|---|---|---|
| ODE 0D (12 estados) | `simulacion/bioconversion_ode.py` | ¿Cuánto CO2/CH4 se generan, cómo crecen las larvas, cómo evolucionan O2, T, RH y los cierres de cámara? |
| CFD 3D (Stokes + transporte pasivo) | `simulacion/cfd_camara.py` | ¿Es válido el supuesto de mezcla homogénea (A1) del ODE en la geometría física de la cámara? |

## 2. Por qué CFD

El ODE asume que el gas de la cabeza de aire está instantáneamente mezclado,
de modo que la lectura del sensor representa el estado de la cámara. El CFD
cuantifica esa hipótesis: inyecta una mancha de trazador pasivo (sin
reacción, sin biología) y mide el tiempo de mezcla τ_mix = cruce de
η(t) = σ_c/c0 por debajo de 5 %. Criterio de aceptación: τ_mix/τ_aire ≤ ~1.2.
El resultado gobierna el diseño de puertos o la corrección de las lecturas
en el gemelo digital (ver `disenho_cfd.md` para el registro de decisiones).

## 3. Entorno de ejecución (reproducción exacta)

- Docker, con imagen derivada: `FROM dolfinx/dolfinx:stable`
  (dolfinx 0.11.0.post0, PETSc con MUMPS, UFL, mpi4py) + `pip install matplotlib gmsh`.
- Construcción (una sola vez):

  docker build -f simulacion/Dockerfile.cfd -t tesis-cfd .

- El script corre dentro del contenedor con el directorio del artículo
  montado en /work. Patrón general:

  docker run --rm --user $(id -u):$(id -g) \
    -e CFD_CONFIG=<etiqueta> \
    -e CFD_IN_Y=<y> -e CFD_IN_Z=<z> -e CFD_TANG=<0|1> \
    -e CFD_OUT_FACE=<x|z> -e CFD_OUT_X=<x> -e CFD_OUT_Y=<y> -e CFD_OUT_Z=<z> \
    -e PYTHONUNBUFFERED=1 -e MPLCONFIGDIR=/tmp/mpl -e HOME=/tmp \
    -v "$BASE":/work -w /work \
    tesis-cfd python3 -u simulacion/cfd_camara.py 2>&1 | grep -v "^Info"

- Utilidades: `-e CFD_MALLA=1` (solo malla + marcado, ~30 s),
  `-e CFD_SOLO_FLUJO=1` (hasta caudales, sin transporte).

## 4. Configuraciones validadas (BCs verificados, etiquetas v2)

| Config | inlet (pared x=0) | tangencial | outlet | CFD_* |
|---|---|---|---|---|
| C1v2 | (0.095, 0.05), normal | no | tapa (0.13, 0.095, Lz), +z | IN_Y=0.095 IN_Z=0.05 TANG=0 OUT_FACE=z OUT_X=0.13 OUT_Y=0.095 |
| C2v2 | (0.05, 0.05), normal | no | pared x=Lx (0.14, 0.13), +x | IN_Y=0.05 IN_Z=0.05 TANG=0 OUT_FACE=x OUT_Y=0.14 OUT_Z=0.13 |
| C4v2 | (0.04, 0.04), a 45° | sí (componente +y) | pared x=Lx (0.14, 0.13), +x | IN_Y=0.04 IN_Z=0.04 TANG=1 OUT_FACE=x OUT_Y=0.14 OUT_Z=0.13 |

Runtime con BCs correctos: ~35–45 min por configuración (Stokes con
no-slip es exigente para MUMPS). Salida válida solo si la línea
`verif BC:` reporta inlet ux = 0.589 y pared ux = 0.

## 5. Salidas (archivado automático por etiqueta CFG)

- `resultados/resumen_<CFG>.md` — métricas y caudales medidos
- `resultados/hist_<CFG>.csv` — historia η(t), masa(t) cada 2 s
- `resultados/cfd_camara_<CFG>.py` — copia exacta del script usado (trazabilidad)
- `imagenes/cfd_tau_mix_<CFG>.pdf` — η(t) semilog + deriva de masa
- `imagenes/cfd_distribucion_<CFG>.pdf` — |u| y c_CO2 final, plano y = Ly/2 (IDW)
- `imagenes/cfd_velocidad_<CFG>.bp`, `cfd_co2_final_<CFG>.bp` — campos para ParaView

Convención de métricas: Q_eff = (|q_in| + q_out)/2 medido en los puertos
(banda de borde del disco discreto da ~15 % bajo el nominal — usar siempre
Q_eff medido); τ_aire = V_aire/Q_eff; índice de bypass = τ_mix/τ_aire y
masa remanente vs. predicción bien mezclada e^(−t/τ_aire).

## 6. Modelo CFD (resumen)

1. Malla: gmsh OCC, caja + discos conformales embebidos en las caras de los
   puertos (fragment; puertos refinados a 1 mm con puntos embebidos;
   bulk ≤ 12 mm). Grupo físico SOLO del volumen (el lector gmsh de dolfinx
   0.11 aborta con facet groups → marcado de facets geométrico en dolfinx).
2. Flujo: Stokes con penalización de divergencia (μ = 1.9e−5 Pa·s,
   λ = 1e4 μ), elementos P2, LU exacto (MUMPS). BCs por subespacios
   (forma canónica de dolfinx 0.11, ver §8).
3. Transporte: advección–difusión (D_CO2 = 2e−5 m²/s) + SUPG,
   Euler implícito (Δt = 2 s, 360 pasos), LU por paso. Trazador inicial:
   c = 1 en z < 0.05. Inlet: c = 0 (Dirichlet); outlet y paredes: flujo
   difusivo nulo natural.

## 7. Modelo ODE (resumen)

14 estados [A, B, L, N, DM, W, Ts, T, w, cCO2, cO2, cCH4, I_T, I_H]:
cinética larval de Eriksen et al. (2022) (compartimentos A, B, L con
rA = a·B, a = amax/(1+(B/Bmax)^α), μB con logistico);
microbioma con cinética de primer orden (k_ref, Q10, Monod en θs);
bаланces de CO2/O2/CH4 unificados para fase abierta y cerrada (cámara
estática: tasa = dC/dt·V); controladores PI de temperatura (Peltier) y
humedad; calendario de cierres adaptativos (Eqs. 22–23 del documento del
modelo). Corre en el Python del host sin dependencias especiales
(numpy, matplotlib): python3 simulacion/bioconversion_ode.py

## 8. Notas críticas de implementación (lecciones que costaron trabajo)

1. **BCs vectoriales en espacios bloqueados**: `fem.dirichletbc` con
   `locate_dofs_topological/geometrical(V, ...)` en dolfinx 0.11 NO aplica
   el valor en los dofs esperados (silencioso). La forma que funciona es
   por componente: `d = fem.locate_dofs_geometrical((V.sub(i), V), marker)`
   (devuelve lista) y `fem.dirichletbc(Constant, d[0], V.sub(i))`.
   Verificación obligatoria post-solve (línea `verif BC`).
   Versiones pre-v2 (C1–C4) usaban la forma incorrecta: sus campos eran
   de fronteras libres y están descartadas; conservadas solo como historia.
2. **Lector gmsh**: `model_to_mesh` requiere grupo físico de volumen y
   falla (abort MPI) con facet groups → marcado de puertos por centroides
   de facets (exacto con discos conformales).
3. **C3 (tangente puro)**: un BC tangente inyecta masa cero — inválido.
   Jet tangencial honesto = componente normal (masa) + tangencial (swirl).
4. **Penalización**: λ = 1e4·μ basta cuando la masa entra/sale por BCs
   Dirichlet; no sirve para "cerrar" el balance por compresibilidad.
5. Un inlet interior (tubo desde tapa) requiere mallar el tubo; la
   aproximación a 45° (C4v2) captura el efecto sin mallarlo.

## 9. Estado

- Script reescrito con BCs verificados y configuración por entorno (v2).
- Pendiente: corridas completas C1v2, C2v2, C4v2 y redacción de
  `contenido_art4.tex`.

## 10. Resultados (BCs verificados, corridas completas)

| Config | Q_eff (L/min) | τ_aire (s) | η(720 s) | τ_mix (s) | τ_mix/τ_aire | masa remanente (720 s) |
|---|---|---|---|---|---|---|
| C1v2 | 0.857 | 175 | 0.246 | > 720 | > 4.1 | 36.5 % |
| C2v2 | 0.899 | 167 | 0.114 | > 720 (cola sugiere ~890) | > 4.3 | 17.2 % |

Ideal bien mezclado: masa remanente a 720 s ≈ 1.3–1.6 %.

Lectura de diseño: en régimen Stokes (laminar puro, sin turbulencia) ninguna
geometría de boquilla en pared alcanza mezcla homogénea: τ_mix supera
ampliamente τ_aire y persiste una fracción grande de volumen muerto (en C1v2
~75 % del volumen efectivamente no ventilado). La configuración diagonal (C2v2)
purga más masa pero no elimina la varianza espacial residual. Pendiente: C4v2
(jet inferior a 45°, componente tangencial) y la discusión contra el Re real
del jet en puerto (~220, transicional), que mezcla más que el Stokes: estos
resultados son cota conservadora inferior de la mezcla real.
