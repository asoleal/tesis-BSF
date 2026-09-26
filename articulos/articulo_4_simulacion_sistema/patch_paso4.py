# patch_paso4.py — PINN a Discusion como trabajo en curso (comprimida)
from pathlib import Path
f = Path("contenido_art4.tex")
s = f.read_text()

# --- 1) quitar §2.3 (capa PINN de metodos) ---
a = "\\subsection{Capa de prediccion con redes informadas por la fisica (PINN)}"
b = "\\section{Simulacion del sistema dinamico}"
assert s.count(a) == 1 and s.count(b) == 1
i0, i1 = s.index(a), s.index(b)
s = s[:i0] + s[i1:]

# --- 2) quitar §3.5 (entrenamiento piloto de resultados) ---
a2 = "\\subsection{Entrenamiento piloto de la capa PINN}"
b2 = "\\section{Resultados y discusion}"
assert s.count(a2) == 1 and s.count(b2) == 1
j0, j1 = s.index(a2), s.index(b2)
s = s[:j0] + s[j1:]

# --- 3) bloque comprimido al final de §4 (antes de Conclusiones) ---
ini = "\\section{Conclusiones}"
assert s.count(ini) == 1
bloque = r'''\subsection{Trabajo en curso: correccion con redes informadas por la fisica}
\label{sec:pinn}

Como trabajo en curso se especifica una capa de correccion basada en una
red neuronal informada por la fisica (PINN), que aprende la discrepancia
entre el gemelo y las observaciones con los residuales del ODE en la
funcion de perdida, de modo que las correcciones respeten los balances de
masa. Tres roles estan definidos: correccion de la tasa de emision a
partir del estado del lecho, inferencia de la biomasa larval no medida
(problema inverso con restricciones DEB) y cuantificacion de CH$_4$ a
partir de la senal no calibrada del TGS2611. Un entrenamiento piloto (red
4-8-1, 49 parametros; pre-entrenamiento con 48 puntos sinteticos tipo
literatura y ajuste fino con las siete tasas netas propias) produce
correcciones $\delta$ entre 0.10 y 0.53 que situan las predicciones dentro
de los rangos observados en los cuatro dias (Figura~\ref{fig:pinn}); la
validacion cruzada queda diferida porque el numero de parametros excede
los datos disponibles. El entrenamiento definitivo, con datos de
literatura digitalizados y los experimentos finales, se reportara por
separado.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=\textwidth]{imagenes/pinn_entrenamiento.pdf}
  \caption{Entrenamiento piloto de la capa PINN. (a) Perdida en el
  pre-entrenamiento (datos sinteticos tipo literatura) y el ajuste fino
  (tasas propias). (b) Paridad prediccion corregida vs observacion.
  (c) Factor de correccion $\delta$ aprendido; las lineas punteadas
  marcan los dias de medicion.}
  \label{fig:pinn}
\end{figure}

Las notas de este trabajo en curso son: (i) los datos de literatura del
pre-entrenamiento son sinteticos ---un sustituto marcado en
\texttt{resultados/pinn\_resumen.md}--- y deben reemplazarse por la
digitalizacion de \textcite{eriksenDynamicModellingFeed2022,
eriksenMetabolicPerformanceFeed2024}; (ii) el componente microbiano
aislado no es identificable con los datos actuales, de modo que $\delta$
corrige la tasa total; (iii) el flujo completo esta versionado en
\texttt{simulacion/pinn\_entrenamiento.py}.

'''
i = s.index(ini)
s = s[:i] + bloque + s[i:]

# --- 4) suavizar item 8 de Conclusiones ---
old8 = (r"  \item El entrenamiento piloto de la capa PINN (red 4-8-1, 49" "\n"
        r"  parametros) con datos sinteticos tipo literatura y siete tasas" "\n"
        r"  propias aprende correcciones $\delta$ entre 0.10 y 0.53 que" "\n"
        r"  situan las predicciones dentro de los rangos observados en los" "\n"
        r"  cuatro dias; la validacion cruzada queda diferida a los" "\n"
        r"  experimentos finales.")
new8 = (r"  \item Como trabajo en curso, una capa de correccion con redes" "\n"
        r"  informadas por la fisica (PINN) muestra en su entrenamiento" "\n"
        r"  piloto (red 4-8-1, 49 parametros; datos sinteticos tipo" "\n"
        r"  literatura y siete tasas propias) correcciones $\delta$ entre" "\n"
        r"  0.10 y 0.53 que situan las predicciones dentro de los rangos" "\n"
        r"  observados; su entrenamiento definitivo queda para los" "\n"
        r"  experimentos finales.")
assert s.count(old8) == 1, s.count(old8)
s = s.replace(old8, new8)

f.write_text(s)
print("OK: paso 4 — PINN comprimida en Discusion, Conclusiones suavizadas")
