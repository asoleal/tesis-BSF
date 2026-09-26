# patch_paso5.py — §3 Resultados por pregunta + §4 Discusion
from pathlib import Path
f = Path("contenido_art4.tex")
s = f.read_text()

# --- A) §2.3: quitar el parrafo de resultados (va a §3.1) ---
a0 = "Un resultado de diseno relevante es que el factor limitante no es $\\Delta$"
a1 = "E5: pico 4604 ppm)."
assert s.count(a0) == 1 and s.count(a1) == 1
i0 = s.index(a0); i1 = s.index(a1) + len(a1)
s = s[:i0].rstrip() + "\n\n" + s[i1:]

# --- B) extraer resultados CFD de §2.4 ---
h1 = "\\subsubsection{Escenarios de puertos en fase ventilada}"
h2 = "\\subsubsection{Mezcla forzada con ventilador en tapa (C5)}"
hs4 = "\\section{Resultados y discusion}"
assert s.count(h1) == 1 and s.count(h2) == 1 and s.count(hs4) == 1
c0 = s.index(h1); c1 = s.index(h2); c2 = s.index(hs4)
cfd_a = s[c0:c1].replace("\\subsubsection{", "\\subsection{")
cfd_b = s[c1:c2].replace("\\subsubsection{", "\\subsection{")
s = s[:c0] + s[c2:]

# --- C) §3: renombrar seccion y recomponer ---
old_sec = "\\section{Simulacion del sistema dinamico}"
assert s.count(old_sec) == 1
s = s.replace(old_sec, "\\section{Resultados}")

e0 = "\\subsection{Escenarios E2--E5}"
r0 = "\\section{Resultados y discusion}"
assert s.count(e0) == 1 and s.count(r0) == 1
j0 = s.index(e0); j1 = s.index(r0)
bloque_viejo = s[j0:j1]

# figura escenarios (extraer del bloque viejo)
lb = "\\label{fig:escenarios}"
assert bloque_viejo.count(lb) == 1
fb0 = bloque_viejo.rindex("\\begin{figure}", 0, bloque_viejo.index(lb))
fb1 = bloque_viejo.index("\\end{figure}", bloque_viejo.index(lb)) + len("\\end{figure}")
fig_esc = bloque_viejo[fb0:fb1]

# validacion (extraer del bloque viejo)
vh = "\\subsection{Validacion contra datos preliminares}"
assert bloque_viejo.count(vh) == 1
val = bloque_viejo[bloque_viejo.index(vh):]
val = val.replace(vh, "\\subsection{Integracion con cria real (P3)}", 1)

nuevo3 = ("\\subsection{Cumplimiento de rango del calendario adaptativo (P1)}"
          "\n\n"
          r"""El calendario de dos pasadas (Seccion 2.3) cumple el tope del
NDIR en los dos escenarios extremos: picos de CO$_2$ de 4978 ppm (E2,
$N_0 = 400$) y 4604 ppm (E5, $N_0 = 700$), con cierres que se acortan desde
30 min al inicio del ciclo hasta 2.5--7 min en el pico de emision y vuelven
a alargarse en la fase de ayuno (Figura~\ref{fig:escenarios}). La
ventilacion de 4 L/min entre cierres mantiene la concentracion de base
entre 500 y 1600 ppm, condicion necesaria: con 1 L/min la base supera los
4300 ppm en el pico y ningun cierre es viable. Las emisiones acumuladas
del ciclo son 0.59 mol de CO$_2$ (E2) y 1.04 mol (E5).

"""
          + fig_esc + "\n\n"
          + cfd_a + cfd_b + "\n"
          + val)

s = s[:j0] + nuevo3 + s[j1:]

# --- D) §4 como Discusion (reescrita; conserva el bloque PINN) ---
d0 = "\\section{Resultados y discusion}"
d1 = "\\section{Conclusiones}"
assert s.count(d0) == 1 and s.count(d1) == 1
k0 = s.index(d0); k1 = s.index(d1)
viejo4 = s[k0:k1]
ph = "\\subsection{Trabajo en curso: correccion con redes informadas por la fisica}"
assert viejo4.count(ph) == 1
pinn_block = viejo4[viejo4.index(ph):]

nueva4 = r'''\section{Discusion}

El veredicto del CFD fija las condiciones de validez del protocolo de
medicion. En fase ventilada las lecturas deben tratarse como estimaciones
acotadas: el sensor promedia una camara con zonas muertas, con sesgo hacia
las concentraciones del entorno del lecho (en C1v2 el 36.5\% del trazador
permanece en la camara a 720 s), de modo que el supuesto A1 no se sostiene
y la pendiente de fase ventilada no es una medicion de precision. En fase
cerrada, en cambio, el ventilador en tapa homogeneiza en menos de 60 s
frente a cierres de 2--30 min: A1 queda validado con un margen de
10--30$\times$ y la pendiente de acumulacion es una medicion de tasa
directa y defendible. El diseno que emerge es: ventilador fijo de material
de baja emision en la tapa, centrado sobre las larvas; medicion del
caudal efectivo de los puertos y de la presion interna de la camara
(deteccion de fugas); cierres estaticos como medicion primaria; y
correccion del volumen muerto del gemelo digital a partir de la fraccion
de mezcla observada.

El switch de prepupa tiene un efecto cuantitativo decisivo sobre el
inventario de GEI del sistema: sin el, el modelo acumula lipidos de forma
ficticia hasta 209 mg/larva y las emisiones de E2 alcanzan 1.37 mol; con
el, la biomasa lipidica final es de 54 mg/larva y las emisiones caen a
0.59 mol. El efecto fija la fraccion de carbono que se emite como CO$_2$
frente a la que se retiene en biomasa de prepupas, magnitud central para
el trabajo de factores de emision que esta instrumentacion habilita.

Las limitaciones se agrupan en dos conjuntos. Instrumentales: $V_{aire}$
esta calculado por geometria y una incertidumbre de $\pm 20\%$ en su valor
se traduce directamente en igual incertidumbre en las tasas estimadas, lo
que justifica su calibracion experimental por dilucion de un trazador; y
el CH$_4$ del TGS2611 es indicativo, con cuantificacion pendiente contra
una referencia. Del lado del modelo: los parametros DEB no han podido
verificarse sin biomasa pesada y el modulo microbiano esta sin
recalibrar (S8), lo que explica la sobreestimacion del pico medio de la
Seccion~\ref{sec:validacion}; ambas quedan resueltas con la campana
experimental final, que ademas pesara biomasa cada dos dias. Con esa
campana, el sistema validado aqui pasa de demostracion a produccion de
factores de emision por dieta.

''' + pinn_block

s = s[:k0] + nueva4 + s[k1:]
f.write_text(s)
print("OK: paso 5 — §3 Resultados por pregunta, §4 Discusion")
