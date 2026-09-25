# patch_s43.py — §4.3 Flujo + §4.4 Transporte (reescritas con ecuaciones)
from pathlib import Path

f = Path("contenido_art4.tex")
s = f.read_text()

ini = "\\subsection{Modelo de flujo}"
fin = "\\subsection{Metricas de mezcla}"
assert s.count(ini) == 1 and s.count(fin) == 1, \
    f"ini={s.count(ini)} fin={s.count(fin)}"
i0, i1 = s.index(ini), s.index(fin)
assert i0 < i1

nuevo = r'''\subsection{Modelo de flujo}

El flujo se resuelve como Stokes estacionario e incompresible,

\begin{equation}
  \mu \nabla^2 \mathbf{u} - \nabla p + \mathbf{f} = 0,
  \qquad \nabla\cdot\mathbf{u} = 0,
\end{equation}

con $\mu = 1.9\times10^{-5}$ Pa·s (aire a 300 K). La incompressibilidad se
impone por penalizacion, $p \approx -\lambda\nabla\cdot\mathbf{u}$ con
$\lambda = 10^4\mu$: la forma debil queda simetrica y definida positiva y
el sistema se resuelve con un solver directo. La fuerza corporal
$\mathbf{f}$ es nula en los escenarios ventilados y vale
$-\chi F_0\,\hat{\mathbf{z}}$ en el escenario C5 de mezcla forzada, donde
$\chi$ es la funcion indicadora de una esfera de radio 2 cm bajo la tapa.

Las condiciones de contorno son de Dirichlet de velocidad: paredes sin
deslizamiento ($\mathbf{u} = 0$) y componente normal $U_{in}$ en cada
puerto (0.589 m/s en el puerto de 3 mm para 1 L/min). Se aplicaron por
subespacios vectoriales y no como condicion vectorial unica: en dolfinx 0.11
un \texttt{Constant} vectorial sobre el espacio bloqueado falla
silenciosamente, de modo que cada componente se restringio por separado y
se verifico tras la solucion (rangos de velocidad en entrada y pared).

La discretizacion usa elementos P2 de Lagrange sobre los $\sim 32\,000$
tetraedros y el sistema lineal se resuelve con PETSc en modo directo
(\texttt{preonly} + LU). La verificacion cuantitativa es el balance de masa
por integrales de frontera: en la configuracion C2 los caudales de entrada
y salida coinciden a 1.034 L/min; el caudal efectivo se define como
$Q_{eff} = (|q_{in}| + q_{out})/2$.

\subsection{Transporte del trazador}

El CO$_2$ se modela como un escalar pasivo $c(\mathbf{x}, t)$ con
adveccion-difusion:

\begin{equation}
  \frac{\partial c}{\partial t} + \mathbf{u}\cdot\nabla c
  - D_{CO_2}\nabla^2 c = 0,
  \qquad D_{CO_2} = 2\times10^{-5}\ \mathrm{m}^2/\mathrm{s},
\end{equation}

la difusion molecular del CO$_2$ en aire a 300 K. La condicion inicial
concentra el trazador en el lecho ($c = 1$ para $z < 0.05$ m, cero en el
resto), lo que reproduce la zona de emision; en fase ventilada la entrada
impone $c = 0$ (aire limpio) y la salida y las paredes toman la condicion
natural de flujo difusivo nulo.

La integracion es de Euler implicito con $\Delta t = 2$ s hasta 720 s, con
elementos P1 de Lagrange y estabilizacion SUPG,

\begin{equation}
  \tau_{SUPG} = \frac{h}{2|\mathbf{u}| + h/\Delta t},
\end{equation}

con $h$ el diametro de celda: el termino estabilizante pondera el residual
fuerte en la direccion del flujo y controla las oscilaciones del P1 puro en
las zonas de alto Peclet. El balance de masa se monitorea paso a paso; la
deriva es inferior al 1\% en la camara sellada.

'''

s = s[:i0] + nuevo + s[i1:]
f.write_text(s)
print("OK: S4.3 y S4.4 escritas,", len(nuevo.splitlines()), "lineas nuevas")
