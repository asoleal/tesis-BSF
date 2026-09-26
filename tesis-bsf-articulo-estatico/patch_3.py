# patch_3.py - nucleo larvario con procedencia por ecuacion + figura TikZ
from pathlib import Path

pm = Path("main.tex"); s = pm.read_text()
old = "\\usepackage{caption}\n"
assert s.count(old) == 1
pm.write_text(s.replace(old, old + "\\usepackage{tikz}\n"))

p = Path("secciones/metodos.tex"); s = p.read_text()
i = s.index("\\subsubsection{Nucleo larvario (DEB de Eriksen)}")
j = s.index("\\subsubsection{Modulo microbiano del lecho}")
NUEVO = r'''\subsubsection{Nucleo larvario (DEB de Eriksen)}

El nucleo sigue el modelo de compartimentos de \textcite{eriksenDynamicModellingFeed2022}: el alimento asimilado entra al compartimento $A$ y de alli se reparte entre crecimiento estructural $B$, reservas lipidicas $L$ y costos metabolicos pagados como CO$_2$ (Figura \ref{fig:nucleo}); los fundamentos DEB de esta particion estan en \parencite{kooijmanDynamicEnergyBudget2009}. A continuacion se indica la procedencia de cada expresion.

\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
  comp/.style={draw, rounded corners, minimum width=2.8cm, minimum height=1.1cm, align=center},
  gas/.style={draw, dashed, rounded corners, minimum width=2.2cm, minimum height=1.1cm, align=center},
  flujo/.style={->, thick},
  etiqueta/.style={midway, fill=white, inner sep=1.5pt, font=\small}]
\node[comp] (alim) at (0,0) {Lecho\\ (alimento)};
\node[comp] (A) at (4.5,0) {$A$: asimilado};
\node[comp] (B) at (9.2,0) {$B$: estructura};
\node[comp] (L) at (9.2,-3.2) {$L$: lipidos};
\node[gas] (CO2) at (4.5,-3.2) {CO$_2$};
\draw[flujo] (alim) -- node[etiqueta]{$r_A = aB(1 - S)$} (A);
\draw[flujo] (A) -- node[etiqueta]{$(1 + Y_B) r_B$} (B);
\draw[flujo] (A) -- node[etiqueta]{$r_{Cm} + Y_B r_B$} (CO2);
\draw[flujo] (A.south east) -- node[etiqueta]{$r_L$} (L.north west);
\draw[flujo] (L) -- node[etiqueta]{$Y_L r_L$} (CO2);
\end{tikzpicture}
\caption{Compartimentos y flujos de carbono del nucleo larvario (adaptado de \parencite{eriksenDynamicModellingFeed2022}). El interruptor de prepupa $S(t)$ (ec. \ref{eq:switch}) cierra la entrada de alimento al final del ciclo; los costos $r_{Cm}$, $Y_B r_B$ y $Y_L r_L$ salen del sistema como CO$_2$.}
\label{fig:nucleo}
\end{figure}

\paragraph{Interruptor de prepupa (componente propio).}
Al final del ciclo larvario, \textit{H. illucens} deja de alimentarse y entra en prepupa, un estadio no alimentario sostenido por las reservas acumuladas \parencite{arreseInsectFatBody2010,llandresDynamicEnergyBudget2015,nayakHermetiaIllucensDiptera2023}. Esto se representa con un interruptor logistico
\begin{equation}
S(t) = \frac{1}{1 + \exp(-(t - t_p)/w_p)},
\label{eq:switch}
\end{equation}
con $t_p$ el dia central y $w_p$ el ancho de la transicion, ajustados al descenso de las tasas medidas en los dias 13 a 17 (sec. \ref{sec:resultados}).

\paragraph{Asimilacion (ecs. 4 y 5 de Eriksen, 2022).}
La tasa especifica de asimilacion decrece con el tamano y el flujo asimilado es proporcional a $B$ \parencite{eriksenDynamicModellingFeed2022}:
\begin{equation}
a(B, t) = a_{\max}\,\frac{1}{1 + (B/B_{\max})^{\alpha}}\,(1 - S(t)),
\qquad r_A = a\,B,
\label{eq:asimilacion}
\end{equation}
donde $B_{\max}$ es la biomasa de referencia y $\alpha$ controla la caida con el tamano. El factor $(1 - S(t))$ es propio: cierra la asimilacion en prepupa.

\paragraph{Crecimiento (ec. 10 de Eriksen, 2022).}
El crecimiento estructural sigue la forma logistica con descuento de mantenimiento de \textcite{eriksenDynamicModellingFeed2022}:
\begin{equation}
G(B) = \max\!\left(1 - (B/B_{\max})^{\beta}, 0\right),
\qquad
\mu_B = \max\!\left(\frac{(a - m)\,G}{1 + Y_B\,G}, 0\right),
\qquad r_B = \mu_B\,B,
\label{eq:crecimiento}
\end{equation}
donde $m$ es la tasa de mantenimiento especifico, en la tradicion de Pirt y los principios macroscopicos de \textcite{roels1980}, y $Y_B$ el costo de crecimiento.

\paragraph{Mantenimiento (Eriksen, 2022, con suelo propio).}
El costo de mantenimiento es proporcional a la estructura, y durante la prepupa cae a un suelo (componente propio) anclado a la medicion de larvas en ayuno del dia 17 (sec. \ref{sec:resultados}):
\begin{equation}
r_{Cm} = m\,B\,\big[(1 - S(t)) + m_{\mathrm{suelo}}\,S(t)\big].
\label{eq:mantenimiento}
\end{equation}

\paragraph{Balance de lipidos y CO$_2$ (ec. 12 de Eriksen, 2022).}
Suponiendo el compartimento $A$ en cuasi-equilibrio ($dA/dt \approx 0$), lo asimilado se reparte instantaneamente entre crecimiento, mantenimiento y reservas, lo que da el flujo neto a lipidos y la produccion larvaria de CO$_2$ como la suma de los tres costos respiratorios \parencite{eriksenDynamicModellingFeed2022,eriksenMetabolicPerformanceFeed2024}:
\begin{equation}
r_L = r_A - (1 + Y_B)\,r_B - r_{Cm},
\qquad
r_{CO_2}^{\mathrm{lar}} = r_{Cm} + Y_B\,r_B + Y_L\,\max(r_L, 0),
\label{eq:co2lar}
\end{equation}
con $Y_L$ el costo de deposito de lipidos; todas las tasas en mg/larva/d.

'''
p.write_text(s[:i] + NUEVO + s[j:])
print("patch_3 aplicado: nucleo larvario reescrito + tikz en preamble")
