# patch_6.py - modulo microbiano con procedencia por ecuacion + figura TikZ
from pathlib import Path
p = Path("secciones/metodos.tex"); s = p.read_text()
i = s.index("\\subsubsection{Modulo microbiano del lecho}")
j = s.index("\\subsubsection", i + 10)
NUEVO = r'''\subsubsection{Modulo microbiano del lecho}

El segundo componente propio describe la degradacion microbiana del lecho: la fraccion no asimilada del sustrato permanece en el lecho y se descompone por actividad microbiana en paralelo al consumo larvario (Figura \ref{fig:lecho}); este modulo es el que permite contrastar el modelo con los controles de alimento solo (sec. \ref{sec:resultados}). A continuacion se indica la procedencia de cada expresion.

\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
  comp/.style={draw, rounded corners=2mm, minimum width=3.0cm, minimum height=1.1cm, align=center, fill=gray!8},
  gas/.style={draw, dashed, rounded corners=2mm, minimum width=2.0cm, minimum height=1.1cm, align=center},
  drv/.style={font=\footnotesize, align=center},
  flujo/.style={->, >=latex, thick, shorten >=3pt, shorten <=3pt},
  mod/.style={->, >=latex, dashed, shorten >=2pt},
  etiqueta/.style={fill=white, inner sep=1.5pt, font=\footnotesize}]
\node[comp] (lecho) at (0,0) {Lecho\\ $D_M$, $W$};
\node[gas] (co2) at (6.5,0) {CO$_2$};
\node[gas] (ch4) at (6.5,-3.4) {CH$_4$};
\node[drv] (T) at (1.6,1.9) {$T$\; ($Q_{10}$)};
\node[drv] (H) at (4.9,1.9) {$\theta_s$\; (Monod)};
\draw[flujo] (lecho) -- node[etiqueta, above=2pt]{$Y_{CO_2}\,k\,D_M$}
                       node[coordinate, pos=0.5] (k) {} (co2);
\draw[flujo] (lecho.south east) -- node[etiqueta]{$Y_{CH_4}\,\xi(\theta_s)\,k\,D_M$} (ch4.north west);
\draw[mod] (T.south) -- (k);
\draw[mod] (H.south) -- (k);
\end{tikzpicture}
\caption{Modulo microbiano del lecho (componente propio). La constante efectiva $k$ (ec. \ref{eq:kmic}) integra la correccion termica $Q_{10}$ y la respuesta de humedad tipo Monod; el CO$_2$ microbiano es proporcional a la materia seca $D_M$ y el CH$_4$ se activa con la fraccion saturada $\theta_s$ via $\xi$ (ec. \ref{eq:rmic}), como indicador cualitativo.}
\label{fig:lecho}
\end{figure}

\paragraph{Humedad del lecho y constante efectiva (formas estandar, combinacion propia).}
El estado de humedad del lecho se resume en la fraccion gravimetrica $\theta_s$ y la constante efectiva combina tres elementos clasicos de la cinetica de descomposicion de residuos \parencite{roels1980,shulerbioprocess2017}: cinetica de primer orden en la materia seca, correccion termica tipo $Q_{10}$ y respuesta de humedad tipo Monod con semisaturacion $K_{\theta}$, acotada por un tope $k_{\max}$; el efecto de la humedad del sustrato sobre el proceso con \textit{H. illucens} esta documentado en \parencite{bekkerImpactSubstrateMoisture2021a,mertenatBlackSoldierFly2019}.
\begin{equation}
\theta_s = \frac{W}{D_M + W},
\qquad
k = \min\!\left(k_{\mathrm{ref}}\,Q_{10}^{(T - T_{\mathrm{ref}})/10}\,
\frac{\theta_s}{\theta_s + K_{\theta}},\; k_{\max}\right),
\label{eq:kmic}
\end{equation}

\paragraph{Tasas de CO$_2$ y CH$_4$ microbianos.}
El CO$_2$ microbiano se obtiene con un rendimiento $Y_{CO_2}$ sobre sustrato degradado, en la tradicion de los balances macroscopicos \parencite{roels1980,shulerbioprocess2017}. El termino de CH$_4$, introducido en este trabajo, modula la tasa con un factor logistico $\xi(\theta_s)$ de la fraccion saturada: la metanogenesis requiere microambientes anoxicos asociados a la saturacion de humedad, y en bioconversion con \textit{H. illucens} se han reportado emisiones de CH$_4$ dependientes del sustrato y las condiciones \parencite{ermolaevGreenhouseGasEmissions2019,parodiBioconversionEfficienciesGreenhouse2020a,boakye-yiadomGreenhouseGasEmissions2022}. Este termino se usa solo como indicador cualitativo, pues el sensor disponible no esta calibrado para tasa neta \parencite{mitchellCalibrationLowCostMethane2024,kiplimoAddressingLowCostMethane2024}.
\begin{equation}
r_{C}^{\mathrm{mic}} = Y_{CO_2}\,k\,D_M,
\qquad
r_{CH_4} = Y_{CH_4}\,\xi(\theta_s)\,k\,D_M,
\qquad
\xi(\theta_s) = \frac{\xi_{\max}}{1 + \exp(-(\theta_s - \theta_{cs})/s_{\theta})},
\label{eq:rmic}
\end{equation}
en g/d.

\paragraph{Balance de masa del lecho.}
\begin{equation}
\frac{dD_M}{dt} = -\frac{N\,r_A}{1000} - k\,D_M,
\qquad
\frac{dW}{dt} = \gamma_w\left(\frac{N\,r_{CO_2}^{\mathrm{lar}}}{1000} + r_C^{\mathrm{mic}}\right),
\label{eq:lecho}
\end{equation}
El primer termino de $dD_M/dt$ es el consumo larvario (con el tope $D_M \geq 0$). El agua del lecho crece con el agua metabolica de la respiracion: $\gamma_w = 0.41$ g de H$_2$O por g de CO$_2$ corresponde a la estequiometria de oxidacion de carbohidratos (1 mol de H$_2$O por mol de CO$_2$, es decir 18/44) \parencite{atkinsPhysicalChemistry2014}; la evaporacion se desprecia por tratarse de una camara cerrada. Cada dos dias, al reponer el alimento, $D_M$ y $W$ se restablecen a 100 y 150 g (250 g, relacion 40/60) como eventos discretos.

'''
p.write_text(s[:i] + NUEVO + s[j:])
print("patch_6 aplicado: modulo microbiano reescrito + figura lecho")
