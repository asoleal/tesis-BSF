# Log de diseño CFD — cámara de bioconversión (artículo 4)

## Objetivo
Verificar el supuesto A1 (mezcla homogénea instantánea) del modelo 0D de 12 estados.
Métricas: η(t) = σ/c0 del trazador CO2 pasivo; τ_mix (cruce η<5%); criterio de
aceptación τ_mix/τ_aire ≤ ~1.2. Flujo nominal Q = 1 L/min, V_aire = 2.5 L,
puertos R = 3 mm, geometría 0.26×0.19×0.18 m.

## Configuraciones evaluadas
- C1: inlet pared x=0 (y=0.095, z=0.05) normal; outlet centro tapa (x=0.13, y=0.095, z=Lz).
- C2: inlet pared x=0 (y=0.05, z=0.05) normal; outlet esquina opuesta (x=Lx, y=Ly-0.05, z=Lz-0.05).
- C3 (INVALIDA): inlet tangente a pared pura (0,U_IN,0). Lección clave: BC tangente
  inyecta masa cero (∫u·n = 0); la bomba de salida succionó sin entrada real
  (caudal in=0.000) y η cayó por artefacto de succión uniforme. Un jet tangencial
  honesto requiere superficie de entrada cuya normal tenga componente en la
  dirección del flujo (tubo), o un vector BC con componente normal + tangencial.
- C4 (propuesta): inlet pared x=0 inferior (0, 0.04, 0.04), u = (U_IN, U_IN, 0)
  → 45° normal/tangente: masa exacta Q por la componente normal, swirl por la
  tangente. Outlet = esquina opuesta superior (igual que C2).

## Resultados válidos
| Config | τ_mix | τ_aire | τ_mix/τ_aire | masa remanente (720 s) |
|---|---|---|---|---|
| C1 | 384 s | 154 s | 2.49 | 18.4 % (ideal: 0.9 %) |
| C2 | 386 s | 145 s | 2.66 | 12.0 % (ideal: 0.8 %) |
| C3 | — inválida (sin inyección de masa) | | | |

Balance de masa verificado: Dirichlet en ambos puertos; C2 in=out=1.034 L/min exacto.

## Conclusiones de diseño
1. En régimen Stokes (laminar), la geometría de boquilla en pared apenas cambia la
   mezcla (relación ~2.5 en C1 y C2): el flujo va por hilos del inlet al outlet y
   el intercambio con zonas muertas es difusión lenta.
2. τ_mix/τ_aire ≈ 2.5 es una COTA CONSERVADORA: el jet real en puerto de 6 mm tiene
   Re ≈ 220 (transicional); la turbulencia real mezclará más que el Stokes. El
   supuesto A1 queda en riesgo pero el error es acotado por estas simulaciones.
3. Bypass cuantificado: 12–18 % del trazador queda atrapado a 720 s vs <1 % ideal.
4. Vías de mejora: (a) jet inferior con componente tangencial (C4, aproxima el tubo
   desde tapa con boquilla horizontal al fondo); (b) aceptar el sesgo y corregir
   la interpretación del sensor en el gemelo digital con τ_mix/fracción de volumen
   muerto del CFD.

## Nota de acoplamiento con el modelo 0D
El CFD no modela biología: es un trazador pasivo que verifica A1. Las variables
objetivo (crecimiento larval A/B/L, CO2 larvario/microbiano, CH4, O2, T, RH) viven
en bioconversion_ode.py. El CFD entrega la corrección de mezcla que el gemelo
digital aplica a las lecturas de sensor (o el criterio de rediseño de cámara).

## Cierre del barrido (C1v2/C2v2/C4v2, BCs verificados)
Tres configuraciones de puertos, todas con τ_mix > 720 s (η no cruza 5 %),
relación τ_mix/τ_aire > 4, fracción de volumen muerto mayoritaria.
El jet a 45° con componente tangencial (C4v2) reproduce C2v2: sin inercia
no hay swirl. Conclusión: en Stokes la mezcla está limitada por difusión
entre líneas de corriente; la geometría de puertos no es la palanca. La
palanca real es el régimen transicional del jet (fuera de Stokes) o mezcla
forzada interna. Ver README §10-11 para la tabla completa.
Nota: falta cfd_distribucion_C1v2.pdf (la corrida terminó tras el tau fig;
regenerable re-corriendo C1v2).
