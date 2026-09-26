# patch_s35_pinn.py — §3.5 entrenamiento piloto + §2.3 rol(i) corregido + item 8
from pathlib import Path
f = Path("contenido_art4.tex")
s = f.read_text()

# --- 1) §2.3: rol (i) sobre tasa total + nota de identificabilidad ---
old_rol = (r"La capa tiene tres roles. (i)~Correccion de las emisiones microbianas" "\n"
           r"(S8): la red toma el estado del lecho (materia seca $DM$, agua $W$," "\n"
           r"temperatura $T_s$, dia del ciclo) y produce un factor $\delta_{mic}$ que" "\n"
           r"modula la tasa microbiana $r_{CO_2}^{mic}$ del modelo; la perdida contrasta" "\n"
           r"$\delta_{mic}\,r_{CO_2}^{mic}$ contra las tasas netas observadas en los" "\n"
           r"cierres.")
new_rol = (r"La capa tiene tres roles. (i)~Correccion de la tasa de emision" "\n"
           r"(S8): la red toma el estado del lecho (materia seca $DM$, agua $W$," "\n"
           r"temperatura $T_s$, dia del ciclo) y produce un factor $\delta$ que" "\n"
           r"modula la tasa total del modelo (larvaria mas microbiana); la" "\n"
           r"perdida contrasta la tasa corregida contra las observadas en los" "\n"
           r"cierres. La correccion microbiana aislada no es identificable con" "\n"
           r"los datos actuales (el termino microbiano de la replica es" "\n"
           r"despreciable frente al larval), de modo que $\delta$ actua sobre la" "\n"
           r"tasa total y la atribucion por componentes queda para la" "\n"
           r"recalibracion del modulo microbiano (Seccion 3.5).")
assert s.count(old_rol) == 1, f"rol={s.count(old_rol)}"
s = s.replace(old_rol, new_rol)

# --- 2) §2.3: nota (i) y conteo de tasas ---
old_n = "(i) reentrenar $\\delta_{mic}$ con las curvas de control\ncompletas;"
new_n = "(i) reentrenar $\\delta$ con las curvas de control completas,\nincluida la descomposicion larvaria/microbiana;"
assert s.count(old_n) == 1, f"nota={s.count(old_n)}"
s = s.replace(old_n, new_n)
assert s.count("(ocho tasas netas del experimento de septiembre)") == 1
s = s.replace("(ocho tasas netas del experimento de septiembre)",
              "(las tasas netas del experimento de septiembre)")

# --- 3) §3.5 nueva subseccion antes de §4 ---
ini = "\\section{Verificacion CFD de la mezcla}"
assert s.count(ini) == 1
nuevo = r'''\subsection{Entrenamiento piloto de la capa PINN}
\label{sec:pinn-piloto}

Se entreno la red del rol (i) (Seccion 2.3) siguiendo el esquema de
transferencia alli descrito. El pre-entrenamiento uso 48 puntos tipo
respirometria larvaria generados sinteticamente en los rangos reportados
para la BSF (correcciones $\delta$ centradas en 1); estos puntos son un
SUSTITUTO explicito de datos de literatura digitalizados y estan marcados
como tales en el repositorio
(\texttt{resultados/pinn\_resumen.md}). El ajuste fino uso las siete
tasas netas propias de los dias 9, 11, 13 y 17 (alimentos D1 y D4).

La red es una MLP 4-8-1 con tangente hiperbolica (49 parametros,
implementada en numpy y entrenada con L-BFGS-B). La funcion de perdida
combina el error de datos con restricciones fisicas:

\begin{equation}
  J = \sum_i \left( \frac{\hat{y}_i - y_i}{\sigma_i} \right)^2
  + \lambda_b \sum_i \mathrm{relu}\left(|g_i| - \ln 10\right)^2
  + \lambda_{bg} \sum_{\mathrm{malla}} \mathrm{relu}\left(|g| - \ln 3\right)^2
  + \lambda_c\, \mathrm{relu}\left(\hat{C} - 2 C_0\right)^2,
\end{equation}

con $\sigma_i = 0.2 y_i + 5$ ppm/min, $g = \ln \delta$ la salida de la
red, $\hat{C}$ el CO$_2$ acumulado del ciclo con la correccion y $C_0$ el
del modelo puro: las dos primeras penalizaciones acotan la correccion
($|\delta| \leq 10$ en los datos y $\leq 3$ fuera del soporte) y la ultima
impone un tope de carbono (la correccion no puede mas que duplicar las
emisiones del ciclo).

El resultado (Figura~\ref{fig:pinn}) es un factor de correccion que decae
de $\delta = 0.53$ en el dia 9 a $\delta = 0.24$ en el dia 13 y alcanza el
piso $\delta = 0.10$ en el ayuno del dia 17. Las tasas corregidas
(112, 140, 31 y 1.5 ppm/min) caen dentro de los rangos observados en los
cuatro dias de medicion, algo que el modelo puro no logra en los dias 11,
13 y 17; las emisiones acumuladas del ciclo pasan de 1.05 a 1.00 mol de
CO$_2$, dentro del tope de carbono impuesto.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=\textwidth]{imagenes/pinn_entrenamiento.pdf}
  \caption{Entrenamiento piloto de la capa PINN. (a) Evolucion de la
  perdida en el pre-entrenamiento (datos sinteticos tipo literatura) y el
  ajuste fino (tasas propias). (b) Paridad entre prediccion corregida y
  observacion. (c) Factor de correccion $\delta$ aprendido frente al dia
  del ciclo; las lineas punteadas marcan los dias de medicion.}
  \label{fig:pinn}
\end{figure}

Tres limitaciones se reportan de forma explicita. Primero, la validacion
cruzada por dia excluido no es aplicable: con 49 parametros y a lo sumo
seis puntos por pliegue, la red interpola el pliegue de entrenamiento y el
error de test no mide generalizacion; queda diferida a los experimentos
finales. Segundo, el componente microbiano aislado no es identificable con
los datos actuales (en la replica el termino microbiano es despreciable
frente al larval), de modo que $\delta$ corrige la tasa total y la
atribucion por componentes queda para la recalibracion del modulo
microbiano (S8). Tercero, los pesos $\lambda$ no se calibraron por
validacion cruzada. En conjunto, el valor de este piloto no es el ajuste
(sesgado por el tamano del conjunto) sino la arquitectura de perdida y el
flujo de entrenamiento, listos para reentrenar cuando lleguen los
experimentos finales.

'''
i = s.index(ini)
s = s[:i] + nuevo + s[i:]

# --- 4) Conclusiones: item 8 ---
assert s.count("\\end{enumerate}") == 1
item8 = (r"  \item El entrenamiento piloto de la capa PINN (red 4-8-1, 49" "\n"
         r"  parametros) con datos sinteticos tipo literatura y siete tasas" "\n"
         r"  propias aprende correcciones $\delta$ entre 0.10 y 0.53 que" "\n"
         r"  situan las predicciones dentro de los rangos observados en los" "\n"
         r"  cuatro dias; la validacion cruzada queda diferida a los" "\n"
         r"  experimentos finales." "\n"
         r"\end{enumerate}")
s = s.replace("\\end{enumerate}", item8)

f.write_text(s)
print("OK: S3.5 agregada, S2.3 ajustada, Conclusiones item 8")
