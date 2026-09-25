p = 'contenido_art4.tex'
s = open(p).read()
if 'sec:dinamica0d' in s:
    print('ya aplicado'); raise SystemExit
i = s.index('\\subsection{Dinamica 0D')
j = s.index('\\subsection{Entradas conmutadas')

nueva = r"""\subsection{Dinamica 0D: doce estados}
\label{sec:dinamica0d}
El gemelo digital tiene doce estados: asimilados y estructura larvaria ($A$, $B$, mg),
lipidos ($L$, mg), poblacion ($N$, larvas), materia seca y agua del lecho ($DM$, $W$, g),
temperaturas del lecho y del aire ($T_s$, $T$, $^{\circ}$C), humedad absoluta del aire
($w$, kg/kg) y concentraciones de CO$_2$, O$_2$ y CH$_4$ del aire interno ($c_i$, mol/m$^3$).
La cinetica larval sigue el modelo de \textcite{eriksenDynamicModellingFeed2022} con
parametros de \textcite{eriksenMetabolicPerformanceFeed2024}; el balance de gases
integra respiracion larvaria, actividad microbiana del lecho y transferencia con el
aire renovado por la ventilacion. El estado se propaga con \texttt{scipy.solve\_ivp}
(metodo LSODA, \texttt{rtol}=10$^{-6}$, \texttt{atol}=10$^{-9}$), con reinicio de
segmentos cuando el protocolo repone alimento (Sec.~\ref{sec:validacion}).

\textbf{Cinetica larval (DEB).} La capacidad de carga estructural decrece
linealmente despues del dia $t_{p,min}$:
\begin{equation}
B_{max}(t) = \left\{ \begin{array}{ll} B_{max,0}, & t \le t_{p,min} \\[2pt]
\max\{B_{max,0} - \rho\,(t - t_{p,min}),\, B_{max,f}\}, & t > t_{p,min}
\end{array} \right.
\end{equation}
Respecto al modelo base se anade el \emph{switch de prepupa} $S_{pr}(t)\in[0,1]$
(supuesto S5, \texttt{supuestos.md}): una logistica centrada en $t_p$ que reproduce
la cesacion de alimentacion observada al final del ciclo:
\begin{equation}
S_{pr}(t) = \frac{1}{1 + e^{-(t - t_p)/s_p}}, \qquad
a(B,t) = \frac{a_{max}}{1 + (B/B_{max}(t))^{\alpha}}\,\bigl(1 - S_{pr}(t)\bigr)
\end{equation}
con $a$ la tasa especifica de asimilacion ($r_A = a\,B$) y $s_p$ el ancho de la
transicion. El crecimiento sigue la regulacion logistica de Eriksen:
\begin{equation}
\mu_B = \left[\frac{(a - m)\,\ell}{1 + Y_B\,\ell}\right]^{+}, \qquad
\ell = \left[1 - \left(\frac{B}{B_{max}(t)}\right)^{\beta}\right]^{+}
\end{equation}
de donde $r_B = \mu_B B$, $r_{CB} = Y_B r_B$, y el mantenimiento, reducido al suelo
de ayuno durante la prepupa, $r_{Cm} = m\,B\,[(1 - S_{pr}) + m_s S_{pr}]$. El pool
$A$ se assume en cuasi-estado estacionario: $r_L = r_A - r_B - r_{Cm} - r_{CB}$.
La produccion de CO$_2$ larval agrupa mantenimiento, sintesis estructural y
sintesis de lipidos ($r_{CL} = Y_L r_L$ si $r_L > 0$):
\begin{equation}
r_{CO_2}^{lar} = r_{Cm} + r_{CB} + r_{CL}, \qquad r_{O_2}^{lar} = r_{CO_2}^{lar}/RQ
\end{equation}

\textbf{Modulo microbiano.} La materia seca del lecho se mineraliza con cinética
de primer orden, correccion $Q_{10}$ y saturacion por humedad (Monod en $\theta_s$,
tope $k_{max}$), y una fraccion anaerobica $\xi$ logistica en $\theta_s$, atenuada
por la actividad bioturbadora de las larvas:
\begin{equation}
k_{mic} = \min\left\{k_{ref}\,Q_{10}^{(T_s - T_{ref})/10}\,
\frac{\theta_s}{\theta_s + K_{\theta}},\, k_{max}\right\}, \quad
\xi = \frac{\xi_{max}}{1 + e^{-(\theta_s - \theta_{cs})/s_{\theta}}}\,
\frac{1}{1 + a_{biot}\,N\,r_A}
\end{equation}
\begin{equation}
r_{CO_2}^{mic} = Y_{CO_2}\,k_{mic}\,DM, \qquad
r_{CH_4} = Y_{CH_4}\,\xi\,k_{mic}\,DM
\end{equation}

\textbf{Balances del lecho y del aire.} El lecho pierde materia seca por ingestion
larval y mineralizacion, con tope de disponibilidad (el consumo cesa si $DM \le 0$),
y agua por evaporacion $E = k_e A_s [\phi(\theta_s) w_{sat}(T_s) - w]^{+}$:
\begin{equation}
\dot{DM} = -N\,r_A/1000 - k_{mic}\,DM, \qquad \dot{W} = -E
\end{equation}
Los gases usan un balance unificado para fase ventilada y cerrada ($\sigma_i = +1$
para CO$_2$/CH$_4$ y $-1$ para O$_2$); en cierre, $Q = 0$ y la tasa es $d c_i/dt = \sigma_i R_i / V_{air}$:
\begin{equation}
V_{air}\,\dot{c}_i = Q\,(c_{in,i} - c_i) + \sigma_i R_i
\end{equation}
La temperatura sigue dos nodos acoplados (lecho y aire): el lecho recibe el calor
metabolico $\gamma_q R_{CO_2}$ y cede al aire; el aire recibe el Peltier
($PI$ sobre $T$, actuador limitado a $\pm 15$ W), pierde por ventilacion y por
$UA$ con el ambiente. La humedad absoluta $w$ balancea evaporacion, agua
metabolica ($\gamma_w R_{CO_2}$) y el humectador ($PI$ sobre $HR$, caudal de
vapor limitado a $5\times 10^{-6}$ kg/s).

\textbf{Condiciones iniciales y parametros.} A(0) = 0.005, $B$(0) = 0.012 y
$L$(0) = 0.003 mg (neonatos), $T_s$(0) = 27, $T$(0) = 28 $^{\circ}$C,
$w$(0) = 0.6\,$w_{sat}$(28), y concentraciones de aire ambiente (420 ppm de
CO$_2$). La racion base es $S_0 = 1.4$ g/larva (40\% materia seca, 60\% agua).
Los valores completos de \texttt{PHY}, \texttt{MIC}, \texttt{FIS} y \texttt{PREP}
constan en \texttt{simulacion/bioconversion\_ode.py} y los supuestos S1--S11 en
\texttt{supuestos.md}. La validacion de esta cinetica frente a medidas de
laboratorio se presenta en la Sec.~\ref{sec:validacion}.

"""
open(p, 'w').write(s[:i] + nueva + s[j:])
print('S2.1 reescrita:', len(nueva), 'chars')
