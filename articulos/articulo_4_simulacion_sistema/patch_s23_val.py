# patch_s23_val.py — agrega §3.4 Validacion contra datos preliminares
from pathlib import Path

f = Path("contenido_art4.tex")
s = f.read_text()

ini = "\\section{Verificacion CFD de la mezcla}"
assert s.count(ini) == 1, f"anchor count={s.count(ini)}"

nuevo = r'''\subsection{Validacion contra datos preliminares}
\label{sec:validacion}

Como primera validacion se uso el experimento de septiembre de 2025: una
panera de plastico sellada (base $30 \times 19$ cm, tapa $36 \times 25$ cm,
altura 17 cm; volumen de tronco de cono $V \approx 12.4$ L, supuesto S1) con
$N = 700$ larvas de \emph{Hermetia illucens} desde huevo, ciclo completo
hasta prepupa (18 d), a temperatura ambiente cercana a 27~$^{\circ}$C (S4).
Se probaron dos alimentos (D1 y D4) a libre disposicion, con 250 g
renovados en cada medicion (S3). El CO$_2$ se midio por NDIR en cierres de
acumulacion; las tasas netas se calculan restando el control del mismo dia
(bandeja con alimento sin larvas, supuesto S7), y el CH$_4$ se uso solo como
indicador (S10). Las tasas extraidas, con su R$^2$ de ajuste, estan en
\texttt{datos/experimentos/datos\_finales\_PINN\_corregidos.csv}.

La Figura~\ref{fig:validacion} contrasta el modelo con las tasas
observadas. El panel (a) muestra la tasa neta de CO$_2$ del modelo (linea)
frente a los valores observados por dia y alimento (D1 circulos, D4
triangulos, ayuno estrella); el panel (b) muestra la biomasa estructural
$B(t)$ y el switch de prepupa $S_{pr}(t)$, con las lineas punteadas de
reposicion de alimento.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=\textwidth]{imagenes/validacion_preliminar.pdf}
  \caption{Validacion preliminar (experimento de septiembre de 2025,
  panera de 12.4 L, $N = 700$). (a) Tasa neta de CO$_2$: modelo (linea) vs.
  observado por dia y alimento. (b) Biomasa estructural $B(t)$ y switch de
  prepupa $S_{pr}(t)$; las lineas punteadas marcan la reposicion de
  alimento.}
  \label{fig:validacion}
\end{figure}

\begin{table}[htbp]
  \centering
  \caption{Tasas netas de CO$_2$ por dia del ciclo: modelo vs.\ rango
  observado (tratamiento $-$ control del mismo dia, ppm/min).}
  \label{tab:validacion}
  \begin{tabular}{c c c}
    \hline
    Dia & Modelo & Observado \\
    \hline
    9  & 210 & 89--309 \\
    11 & 289 & 124--165 \\
    13 & 129 & 22--103 \\
    17 & 15  & 0--14 \\
    \hline
  \end{tabular}
\end{table}

El modelo reproduce el orden de magnitud y la caida final hacia el ayuno de
prepupa (15 ppm/min frente a 0--14 observados en el dia 17), pero
sobreestima el pico del ciclo medio (289 frente a 124--165 en el dia 11).
La causa principal es que los parametros DEB (supuesto S6) no han podido
verificarse sin biomasa larval medida; en consecuencia, la amplitud del pico
queda sujeta a la calibracion con los experimentos finales. La medicion de
ayuno del dia 17 (14.3 ppm/min) fue precisamente la evidencia que motivo el
switch de prepupa (S5). Otras limitaciones abiertas: el modulo microbiano
sin recalibrar frente a las curvas de control (S8), $V_{aire}$ sujeto a
calibracion por trazador, y CH$_4$ no cuantitativo (S10).

'''

i = s.index(ini)
s = s[:i] + nuevo + s[i:]
f.write_text(s)
print("OK: S3.4 validacion agregada,", len(nuevo.splitlines()), "lineas nuevas")
