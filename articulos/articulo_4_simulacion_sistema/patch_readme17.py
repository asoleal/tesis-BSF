# patch_readme17.py — agrega §17 al README (plan de publicaciones + reorganizacion)
from pathlib import Path
f = Path("README.md")
s = f.read_text()
assert "## 17." not in s, "§17 ya existe"

s17 = """
## 17. Plan de publicaciones y reorganizacion (2026-09-26, decisiones)

### Diagnostico
El articulo actual es material de nivel 4 pero disperso: objetivo implicito,
experimento referenciado sin definir, 4 hilos (gemelo, CFD, validacion, PINN)
sin pregunta unica. Evaluacion por campo (1-5): tecnologico 4, matematica
aplicada 3, ambiental 2, zootecnia 1.

### Decision tomada
Reenfocar el ARTICULO 1 al campo tecnologico (sistema de medicion como
protagonista, gemelo como consumidor de datos, PINN a discusion/trabajo
futuro). El campo ambiental (nucleo del doctorado) queda para el Articulo 2,
con los experimentos finales.

### Portafolio (6 papers, maximo 2 activos a la vez)
1. **A1 (ahora, tecnologico):** sistema de medicion de CO2 en contenedor BSF:
   cierres adaptativos + CFD + gemelo + demo sep 2025. Ya escrito; se
   reorganiza (ver tareas abajo).
2. **A2 (ambiental, mes 1-4):** GEI en bioconversion con H. illucens: factores
   de emision por dieta (g CO2-eq/kg sustrato y /kg biomasa), reparticion de
   carbono, comparacion vs compostaje. Requiere: CH4 calibrado, V_air por
   trazador, replicas.
3. **A3 (zootecnia, mes 3-4):** modelo de crecimiento DEB + switch de prepupa
   validado con biometria (pesaje cada 2 d). JIFS o RCCP. Requiere biomasa
   pesada.
4. **A4 (capstone, mes 5+):** gemelo digital + PINN entrenada con datos
   reales + dashboard ("sistema inteligente"). Absorbe §2.3/§3.5 actuales.
5. **A5 (opcional, ambiental):** factores de emision por TIPO de alimento
   (subconjunto del DOE de A2).
6. **A6 (opcional, zootecnia):** lineas/semillas de huevo >= 2 fuentes;
   SIN >=2 fuentes este paper NO sale.

### Regla de diseno experimental (critica)
Una sola campana extendida (1-2 meses) alimenta A2 y A3 simultaneamente.
Disenar el DOE completo ANTES de iniciar: dietas x lineas x replicas x
pesajes cada 2 d + mediciones de gas con el protocolo de A1. Cada paper toma
su contraste sin solaparse (evitar salami slicing).

### Tareas de reorganizacion de A1 (en orden)
1. Titulo nuevo + Resumen + Introduccion con objetivo explicito y 3 preguntas
   (P1 calendario/NDIR, P2 mezcla A1/CFD, P3 gemelo vs observado).
2. Nueva §2.1 "Sistema experimental" (panera, alimentos D1/D4, sensores,
   protocolo dias 9/11/13/17, controles _alimento, 250 g ad libitum) —
   arregla el "experimento sin definir".
3. §2.1 actual (ODE) comprimida y renumerada; §2.2 cierres -> 2.3; CFD
   integra como 2.4.
4. PINN (§2.3 + §3.5) movida a Discusion como trabajo en curso (parrafo +
   notas), liberando foco.
5. Resultados espejando P1-P3; Discusion; Conclusiones una por pregunta.
6. Pendientes menores: cosmeticos (Fig 2 width, caveat Fig 3b al caption).

### Estado
- Pasos 1-6 de A1: PENDIENTE (parches por paso, uno a la vez).
- Campana de experimentos: por disenar (DOE).
"""

f.write_text(s + s17)
print("OK: README §17 agregada")
