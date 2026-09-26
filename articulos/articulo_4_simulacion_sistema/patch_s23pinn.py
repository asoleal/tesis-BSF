from pathlib import Path
f = Path("contenido_art4.tex")
s = f.read_text()

ini = "\\section{Simulacion del sistema dinamico}"
assert s.count(ini) == 1, s.count(ini)

nuevo = r'''\subsection{Capa de prediccion con redes informadas por la fisica (PINN)}
\label{sec:pinn}

El gemelo mecanicista tiene errores estructurales conocidos y senalados
(S6, S8, S10): no es realista esperar que el ajuste fino de parametros los
absorba todos. Se plantea entonces una capa de correccion basada en una red
neuronal informada por la fisica (PINN, por sus siglas en ingles): una red
pequena que aprende la discrepancia entre el modelo y las observaciones con
los residuales del propio ODE (Seccion 2.1) en la funcion de perdida, de
modo que las correcciones respeten los balances de masa y la cinetica DEB.
La red se mantiene deliberadamente pequena: con los datos disponibles
(ocho tasas netas del experimento de septiembre) una arquitectura grande
sobreajustaria; el mecanismo de generalizacion es la fisica, no el numero
de parametros.

La capa tiene tres roles. (i)~Correccion de las emisiones microbianas
(S8): la red toma el estado del lecho (materia seca $DM$, agua $W$,
temperatura $T_s$, dia del ciclo) y produce un factor $\delta_{mic}$ que
modula la tasa microbiana $r_{CO_2}^{mic}$ del modelo; la perdida contrasta
$\delta_{mic}\,r_{CO_2}^{mic}$ contra las tasas netas observadas en los
cierres. (ii)~Inferencia de la biomasa larval no medida (S6): un problema
inverso en el que la red propone trayectorias $B(t)$ condicionadas por las
observaciones de CO$_2$ y penalizadas por los residuales DEB; la
incertidumbre se estima con un conjunto pequeno de redes entrenadas con
semillas distintas. (iii)~Cuantificacion de CH$_4$ (S10): la senal no
cuantitativa del sensor TGS2611 se fusiona con las variables ambientales
bajo el balance de masa del ODE para producir una tasa calibrada de CH$_4$.

El entrenamiento es por transferencia: pre-entrenamiento con datos de
respirometria larvaria reportados en la literatura
\parencite{eriksenDynamicModellingFeed2022,
eriksenMetabolicPerformanceFeed2024} y ajuste fino con los datos propios;
la validacion es por dia excluido (leave-one-day-out), coherente con el
tamano del conjunto.

\textbf{Notas pendientes.} Esta capa se especifica en el presente articulo
y su entrenamiento completo queda para la fase de experimentos finales: con
$ocho$ observaciones la red solo puede fijar unos pocos grados de libertad
y se espera un ajuste pobre en esta etapa; lo relevante aqui es la
arquitectura de perdida, no el ajuste. Cuando lleguen los experimentos
finales habra que: (i) reentrenar $\delta_{mic}$ con las curvas de control
completas; (ii) verificar la $B(t)$ inferida contra biomasa pesada;
(iii) recalibrar CH$_4$ contra una referencia; y (iv) fijar el peso de los
residuales del ODE en la perdida por validacion cruzada.

'''

i = s.index(ini)
s = s[:i] + nuevo + s[i:]
f.write_text(s)
print("OK: S2.3 PINN agregada,", len(nuevo.splitlines()), "lineas nuevas")
