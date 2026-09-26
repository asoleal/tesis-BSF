# patch_8.py - balance de camara con procedencia por ecuacion + figura TikZ
from pathlib import Path
p = Path("secciones/metodos.tex"); s = p.read_text()
i = s.index("\\subsubsection{Balance de gas en camara estatica}")
j = s.index("\\begin{table}", i)
NUEVO = r'''\subsubsection{Balance de gas en camara estatica}

El tercer componente conecta las fuentes con la medicion: durante un cierre, el CO$_2$ y el CH$_4$ se acumulan en el volumen de aire de la camara y el sensor registra la pendiente de concentracion (Figura \ref{fig:camara}).

\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
  comp/.style={draw, rounded corners=2mm, minimum width=2.6cm, minimum height=1.1cm, align=center, fill=gray!8},
  inst/.style={draw, rounded corners=2mm, minimum width=2.6cm, minimum height=1.1cm, align=center},
  flujo/.style={->, >=latex, thick, shorten >=3pt, shorten <=3pt},
  etiqueta/.style={fill=white, inner sep=1.5pt, font=\footnotesize}]
\node[comp] (larvas) at (0,1.8) {Larvas\\ $N$};
\node[comp] (lecho) at (0,-1.8) {Lecho\\ $D_M$, $W$};
\node[comp] (camara) at (6.4,0) {Camara cerrada\\ $V_{\mathrm{aire}}$};
\node[inst] (sensor) at (12.2,0) {Sensor\\ pendiente $dc/dt$};
\draw[flujo] (larvas.east) -- node[etiqueta, sloped, above]{$N\,r_{CO_2}^{\mathrm{lar}}$} (camara.170);
\draw[flujo] (lecho.east) -- node[etiqueta, sloped, below]{$r_C^{\mathrm{mic}}$} (camara.190);
\draw[flujo] (camara) -- node[etiqueta, above=2pt]{$R_{CO_2}/V_{\mathrm{aire}}$} (sensor);
\end{tikzpicture}
\caption{Balance de gas en la camara estatica, sobre el metodo estandar \parencite{hutchinsonMethodsSoilAnalysis1992}. Las fuentes larvaria y microbiana acumulan gas en el volumen de aire durante un cierre (ec. \ref{eq:camara}); el sensor registra la pendiente, llevada a ppm/min con la ley de gas ideal (ec. \ref{eq:ppm}).}
\label{fig:camara}
\end{figure}

\paragraph{Acumulacion en el cierre (balance propio sobre el metodo estandar).}
El metodo de camara estatica estima la tasa a partir de la acumulacion lineal del gas en un volumen conocido \parencite{hutchinsonMethodsSoilAnalysis1992}; con las fuentes del modelo, el balance molar sobre el aire de la camara es
\begin{equation}
\frac{dc_{CO_2}}{dt} = \frac{R_{CO_2}}{V_{\mathrm{aire}}},
\qquad
R_{CO_2} = \frac{N\,r_{CO_2}^{\mathrm{lar}} + 1000\,r_C^{\mathrm{mic}}}{44000},
\label{eq:camara}
\end{equation}
con $R_{CO_2}$ en mol/d: el termino larvario pasa de mg a g y ambos a mol con la masa molar del CO$_2$ (44 g/mol); un balance analogo aplica para $c_{CH_4}$ con $R_{CH_4} = 1000\,r_{CH_4}/16000$ (16 g/mol).

\paragraph{Escala del sensor (ppm/min).}
La comparacion con el sensor se hace en la escala propia del metodo \parencite{hutchinsonMethodsSoilAnalysis1992}, convirtiendo la acumulacion molar a ppm/min con la ley de gas ideal \parencite{atkinsPhysicalChemistry2014}:
\begin{equation}
r_{\mathrm{ppm}} = \frac{R_{CO_2}}{V_{\mathrm{aire}}}\,\frac{R_g\,T}{P}\,\frac{10^6}{1440}
\quad [\mathrm{ppm/min}],
\label{eq:ppm}
\end{equation}
con $R_g = 8.314$ J/(mol$\cdot$K) y $T$, $P$ la temperatura y presion de la camara: el factor $R_g T/P$ es el volumen molar, $10^6$ lleva a ppm y 1440 de dias a minutos.

\paragraph{Validacion por descomposicion de fuentes (estrategia propia).}
La validacion explota la descomposicion de fuentes: la componente larvaria $N\,r_{CO_2}^{\mathrm{lar}}$ se compara con la tasa neta medida (tratamiento menos control del mismo dia, promedio de tres repeticiones), y la componente microbiana $r_C^{\mathrm{mic}}$ con las curvas de control (alimento solo). Los parametros del nucleo DEB se tomaron de \textcite{eriksenDynamicModellingFeed2022,eriksenMetabolicPerformanceFeed2024} (Cuadro \ref{tab:parametros}); los del modulo microbiano ($k_{\mathrm{ref}}$, $Y_{CO_2}$) se calibran en este trabajo contra las curvas de control (sec. \ref{sec:resultados}).

'''
p.write_text(s[:i] + NUEVO + s[j:])
print("patch_8 aplicado: balance de camara reescrito + figura camara")
