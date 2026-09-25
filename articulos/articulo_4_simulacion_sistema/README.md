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

| C4v2 | 0.845 | 177 | 0.110 | > 720 | > 4.1 | 18.5 % |

Conclusión del barrido C1v2/C2v2/C4v2 (BCs verificados): en régimen Stokes
(laminar puro) la geometría de boquilla en pared NO logra mezcla homogénea —
la componente tangencial (C4v2) no genera swirl sin inercia. τ_mix/τ_aire > 4
es cota conservadora inferior: la mezcla real del jet transicional (Re ~ 220
en puerto de 6 mm) será mayor. Implicaciones: (1) validación experimental de
mezcla en el piloto (sensores a dos alturas); (2) corrección de volumen
muerto en el gemelo digital con las fracciones medidas aquí; (3) si se
requiere mezcla garantizada: ventilador interno o jet turbulento — no
alcanzable con geometría de puertos en laminar.

## 12. C5 — Ventilador en tapa (configuración final de diseño)

> Actualiza la conclusión del §11: el ventilador ya no es solo "recomendado",
> queda **validado** como garantía del supuesto A1 en cámara sellada.

### Modelo del fan
- Fuente de momentum (fuerza corporal en −z) dentro de una esfera de r = 0.02 m
  centrada en (Lx/2, Ly/2, Lz − 0.03): bajo la tapa, directamente sobre las larvas.
  No se mallan las aspas (modelo estándar de ventilador como actuador de momentum).
- Variables de entorno nuevas en `cfd_camara.py`:
  `CFD_FAN` (0/1), `CFD_FAN_F` (fuerza N/m³, default 400), `CFD_FAN_R` (radio),
  `CFD_Q0` (1 = cámara sellada, U_IN = 0 en ambos puertos), `CFD_NPASOS`.

### Resultados (cámara sellada, mezcla solo por fan)

| Config        | F0 (N/m³) | \|u\| max (m/s) | τ_mix  | deriva masa | η(t=62 s) |
|---------------|-----------|-----------------|--------|-------------|-----------|
| C5cerrada     | 400       | 1101.8          | 4 s    | −1.00 %     | 0.000     |
| C5cerrada_F4  | 4         | 11.02           | 24 s   | −0.61 %     | 0.000     |

- **Escalamiento verificado**: Stokes es lineal → |u| ∝ F0 exacto
  (100× menos fuerza = 100× menos velocidad: 1101.8 → 11.018).
- **La mezcla satura**: al bajar F0 100×, τ_mix solo sube 4 → 24 s. En ambos
  casos η < 0.05 % antes de t = 60 s.

### Conclusión de diseño
- Con cualquier recirculación significativa del fan, la cámara sellada se
  homogeniza en **< 60 s**, frente a cierres de medición de 10–30 min:
  el supuesto A1 (bien mezclado) queda garantizado en la fase estática con
  margen de 10–30×.
- Stokes sigue siendo cota conservadora: con un fan real de 40 mm (Re ~ 10³,
  flujo transicional/turbulento) la mezcla real es más rápida que el límite laminar.
- Velocidad fija, no variable: la mezcla saturada hace innecesario el control
  de velocidad. Especificación: fan axial 40 mm 5 V centrado en tapa, material
  de baja emisión (evitar COV/plásticos baratos que contaminen las medidas).

### Reproducción (cámara sellada + fan)

    docker run --rm --user $(id -u):$(id -g) \
      -e CFD_CONFIG=C5cerrada_F4 -e CFD_Q0=1 -e CFD_FAN=1 -e CFD_FAN_F=4 \
      -e CFD_NPASOS=240 -e CFD_OUT_FACE=x -e CFD_OUT_Y=0.14 -e CFD_OUT_Z=0.13 \
      -e CFD_IN_Y=0.05 -e CFD_IN_Z=0.05 \
      -e PYTHONUNBUFFERED=1 -e MPLCONFIGDIR=/tmp/mpl -e HOME=/tmp \
      -v "$BASE":/work -w /work tesis-cfd python3 -u simulacion/cfd_camara.py

## 13. Validación preliminar con experimentos reales (sep 2025) — actualiza §7 y §9

Datos: `../../datos/experimentos` (carpetas 9/11/13/17 = **días del ciclo**, no solo
fechas; D1/D4 = dos alimentos; `*_alimento` = controles solo-alimento 250 g;
`Larvas_Ayuno` = larvas sin alimento, día 17). Panera sellada 30×19 → 36×25 × 17 cm
(V_air ≈ 12.1 L), N = 700, alimento repuesto en cada medición (ad libitum).
Tasas ya extraídas en `datos_finales_PINN_corregidos.csv` (ppm/min, con R²);
protocolo de resta: `restar_controles.py` (tratamiento − control del mismo día).

Cruce modelo-vs-dato (conversión a ppm/min a 12.1 L):
- **Días 9–11: el modelo acierta** — predice 226–417 ppm/min netos; observado 88–650
  según alimento/estado. El caso de saturación en ~6 min es el experimento 1
  (`experimento1_D1_voraz_tenian_hambre`): burst de día 9 ≈ pico del modelo.
- **Días 13–17: fallaba por mecanismo faltante, no por parámetros** — con alimento
  disponible, el descenso observado (→ ~0 en día 17) es cesación por prepupa.
  El modelo no la tenía y mantenía 489 ppm/min. → switch PREP (§14).
- **Módulo microbiano subestima el CO2 del alimento 5–20×** (modelo 0.6–2 vs
  controles 26–110 ppm/min). Recalibración pendiente (guía §15.6).
- **CH4 no es cuantitativo** (TGS2611; pasos de 200 ppm; controles "saturados"):
  indicador solamente, no entra a ajuste ni PINN hasta calibrar.

Supuestos operativos S1–S11: ver `supuestos.md` (cada uno con su parámetro en el código).

## 14. Cambios al ODE (commits fb2e845, edcb5ae + ventilación S11)

1. **PREP = dict(t_p=12.0, ancho=1.0, m_suelo=0.13)** (S5): la asimilación `a` se
   multiplica por (1 − S) con S logística centrada en día 12; el mantenimiento
   cae a ×0.13 (suelo 0.64 mg/larva/d, anclado a la medición de ayuno día 17).
   Colateral: elimina la acumulación fantasma de lípidos (L final 209 → 54 mg) que
   inflaba las emisiones ~2.3× (E2: 1.37 → 0.59 mol).
2. **Tope de ingestión**: `comida = 0 si DM ≤ 0` — el lecho ya no se vuelve negativo.
3. **Calendario adaptativo en dos pasadas** (`calendario_adaptativo`):
   Δ_k = margen / ppm_s, con la tasa evaluada en el **punto medio** del cierre
   (orden 2), piso 120 s, techo 1800 s; margen = 5000 − c(t_ck) de la pasada 1.
4. **Ventilación entre cierres 4 L/min** (`caudal()` devuelve 6.6667e-5, ambas
   bombas). CRÍTICO: con 1 L/min la base estacionaria entre cierres (c_in + R/Q)
   sube a ~4300 ppm en el pico del ciclo y **ningún** cierre queda bajo 5000 ppm
   aun con Δ en el piso — el constraint de diseño es la ventilación, no el Δ.
   Con 4 L/min la base queda 500–1600 ppm.

Verificado (E2/E5, este README §7): picos 4978/4604 ppm ≤ 5000; Δ_k van de
30 min (días 3–6) a 2.5–5 min (pico) a 15–27 min (prepupa).
Reproducción: `python3 simulacion/bioconversion_ode.py` (host: numpy/scipy/matplotlib).

## 15. GUÍA — qué cambiar cuando lleguen los experimentos finales

| # | Dato nuevo del experimento final | Dónde se cambia | Cómo |
|---|---|---|---|
| 1 | V_air medido (llenar cámara de agua) | `FIS['Vair']` | 12.1e-3 (panera) o el valor medido; re-verificar picos con el run estándar |
| 2 | N real y razón de alimento | `correr(N0, ...)` y `S0 = 1.4*N0` | sustituir 1.4 g/larva por la razón medida; si hay agotamiento, quitar ad libitum (el tope DM ≥ 0 ya existe) |
| 3 | T de proceso controlada | `FIS['Ta']`, `FIS['Tref']` | valor del nuevo protocolo |
| 4 | Edad de prepupa observada | `PREP['t_p']`, `PREP['ancho']` | ajustar con el descenso de tasas; `m_suelo` con un test de ayuno |
| 5 | Biomasa larval (pesadas) | `PHY` (YB, YL, m, amax, Bmax0) | mínimos cuadrados de B(t) medido vs simulado; validar S6 |
| 6 | Controles solo-alimento nuevos | `MIC` (kref, kmax, YCO2, Kth, th_cs) | extraer tasas como en `datos_finales_PINN_corregidos.csv` y minimizar error en ppm/min a V_air real; valida S8 |
| 7 | Rango del NDIR usado | `cmax` en `calendario_adaptativo` | 5000 (S8) o 10000 (K30) |
| 8 | CH4 calibrado (p.ej. Mitchell 2024) | `YCH4`, `xi_*` en MIC + texto | recién entonces incluir CH4 en el ajuste |
| 9 | Nuevos días de validación | escenario réplica (Fase 2, pendiente) | generalizar el calendario de cierres a las edades medidas |
| 10 | Texto del artículo | `contenido_art4.tex` | §2 (switch + microbiano), §3 nueva subsección de validación, §5 (ventilación como constraint), §6, anexo de parámetros |

Estado de figuras: `imagenes/sim_E2.pdf`, `sim_E5.pdf` regenerados con la dinámica
con prepupa; `esquema_sistema.pdf` v3. Pendiente Fase 2: script de réplica del
experimento (panera) + figura modelo-vs-datos; Fase 3: texto (ítem 10).
