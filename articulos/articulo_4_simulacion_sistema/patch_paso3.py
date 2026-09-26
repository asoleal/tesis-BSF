# patch_paso3.py — §2 "Materiales y metodos": ODE comprimida + CFD integrada
from pathlib import Path
f = Path("contenido_art4.tex")
s = f.read_text()

# --- 1) renombrar seccion ---
old = "\\section{Modelo del sistema}"
assert s.count(old) == 1
s = s.replace(old, "\\section{Materiales y metodos}")

# --- 2) ODE comprimida ---
a = "\\subsection{Dinamica 0D: doce estados}"
b = "\\subsection{Entradas conmutadas y cierres}"
assert s.count(a) == 1 and s.count(b) == 1
i0, i1 = s.index(a), s.index(b)
nueva_ode = r'''\subsection{Dinamica 0D: doce estados}
\label{sec:dinamica0d}

El gemelo digital tiene doce estados: asimilado y estructura larvaria
($A$, $B$, mg), lipidos ($L$, mg), poblacion ($N$, larvas), materia seca y
agua del lecho ($DM$, $W$, g), temperaturas del lecho y del aire ($T_s$,
$T$, $^{\circ}$C), humedad absoluta del aire ($w$, kg/kg) y concentraciones
de CO$_2$, O$_2$ y CH$_4$ del aire interno ($c_i$, mol/m$^3$). La cinetica
larval sigue el modelo DEB de \textcite{eriksenDynamicModellingFeed2022}
con parametros de \parencite{eriksenMetabolicPerformanceFeed2024}: la
capacidad de carga estructural decrece linealmente despues del dia
$t_{p,min}$ (inicio de la prepupa),

\begin{equation}
  B_{max}(t) = \left\{ \begin{array}{ll} B_{max,0}, & t \leq t_{p,min} \\
  \max\{B_{max,0} - \rho\,(t - t_{p,min}),\, B_{max,f}\}, & t > t_{p,min}
  \end{array} \right.
\end{equation}

y respecto al modelo base se anade el switch de prepupa $S_{pr}(t) \in [0;1]$
(supuesto S5), una logistica centrada en $t_p$ que reproduce la cesacion
de alimentacion observada al final del ciclo:

\begin{equation}
  S_{pr}(t) = \frac{1}{1 + e^{-(t-t_p)/s_p}}, \qquad
  a(B,t) = \frac{a_{max}}{1 + (B/B_{max}(t))^{\alpha}} (1 - S_{pr}(t))
\end{equation}

con $a$ la tasa especifica de asimilacion ($r_A = aB$) y $s_p$ el ancho de
la transicion. El crecimiento sigue la regulacion logistica de Eriksen:

\begin{equation}
  \mu_B = \left[ \frac{(a - m)\ell}{1 + Y_B\,\ell} \right]^{+}, \qquad
  \ell = \left[ 1 - \left( \frac{B}{B_{max}(t)} \right)^{\beta} \right]^{+}
\end{equation}

de donde $r_B = \mu_B B$, $r_{CB} = Y_B r_B$, y el mantenimiento, reducido
al suelo de ayuno durante la prepupa, $r_{Cm} = mB[(1-S_{pr}) + m_s S_{pr}]$.
La produccion de CO$_2$ larval agrupa mantenimiento, sintesis estructural
y sintesis de lipidos ($r_{CL} = Y_L r_L$ si $r_L > 0$):

\begin{equation}
  r^{lar}_{CO_2} = r_{Cm} + r_{CB} + r_{CL}, \qquad
  r^{lar}_{O_2} = r^{lar}_{CO_2}/RQ
\end{equation}

El lecho mineraliza materia seca con cinetica de primer orden, correccion
$Q_{10}$ y saturacion por humedad, con una fraccion anaerobica atenuada por
la actividad bioturbadora de las larvas (modulo microbiano, supuesto S8,
pendiente de recalibracion); pierde materia por ingestion larval, con tope
de disponibilidad, y agua por evaporacion. Los gases usan un balance
unificado para fase ventilada y cerrada ($\sigma_i = +1$ para CO$_2$/CH$_4$
y $-1$ para O$_2$); en cierre, $Q = 0$ y la tasa es
$\mathrm{d}c_i/\mathrm{d}t = \sigma_i R_i/V_{air}$:

\begin{equation}
  V_{air}\,\dot{c}_i = Q\,(c_{in,i} - c_i) + \sigma_i R_i
\end{equation}

La temperatura sigue dos nodos acoplados (lecho y aire) con calor
metabolico y Peltier, y la humedad se balancea con evaporacion; los
detalles y los parametros completos (PHY, MIC, FIS, PREP) estan en
\texttt{simulacion/bioconversion\_ode.py} y los supuestos S1--S11 en
\texttt{supuestos.md}. Las condiciones iniciales replican el experimento:
$A(0) = 0.005$, $B(0) = 0.012$ y $L(0) = 0.003$ mg (neonatos), $T_s(0) = 27$,
$T(0) = 28$~$^{\circ}$C, $w(0) = 0.6\,w_{sat}(28)$, concentraciones de aire
ambiente (420 ppm de CO$_2$) y razon base de alimento $S_0 = 1.4$ g/larva
(40\% materia seca, 60\% agua). La implementacion usa
\texttt{scipy.solve\_ivp} (LSODA, rtol = $10^{-6}$), con reinicio de
segmentos en cada reposicion de alimento; el codigo, las figuras y los
resultados estan versionados en el repositorio del proyecto. La validacion
de esta cinetica frente a medidas de laboratorio se presenta en la
Seccion~\ref{sec:validacion}.

'''
s = s[:i0] + nueva_ode + s[i1:]

# --- 3) CFD: de \section a \subsection dentro de §2 ---
c0 = "\\section{Verificacion CFD de la mezcla}"
c1 = "\\section{Resultados y discusion}"
assert s.count(c0) == 1 and s.count(c1) == 1
j0, j1 = s.index(c0), s.index(c1)
cfd = s[j0:j1]
cfd = cfd.replace("\\section{Verificacion CFD de la mezcla}",
                  "\\subsection{Verificacion CFD de la mezcla}")
assert cfd.count("\\subsection{") >= 8
cfd = cfd.replace("\\subsection{", "\\subsubsection{")
s = s[:j0] + s[j1:]                     # quitar de su posicion
sim = "\\section{Simulacion del sistema dinamico}"
assert s.count(sim) == 1
i = s.index(sim)
s = s[:i] + cfd + s[i:]                 # insertar antes de Simulacion

# --- 4) absorber "Implementacion" en la ODE (ya cubierto arriba) ---
imp = "\\subsection{Implementacion}"
esc = "\\subsection{Escenarios E2--E5}"
assert s.count(imp) == 1 and s.count(esc) == 1
k0, k1 = s.index(imp), s.index(esc)
s = s[:k0] + s[k1:]

f.write_text(s)
print("OK: paso 3 — §2 metodos completa, CFD integrada, ODE comprimida")
