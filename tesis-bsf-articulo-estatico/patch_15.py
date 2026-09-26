# patch_15.py - metodo numerico justificado + supuestos de la solucion
from pathlib import Path
p = Path("secciones/metodos.tex"); s = p.read_text()

old1 = ("Cada segmento entre eventos se integro con \\texttt{solve\\_ivp} (metodo LSODA, "
"tolerancias relativa $10^{-6}$ y absoluta $10^{-9}$), implementado en Python 3 con NumPy y SciPy.")
new1 = r'''Cada segmento entre eventos se integro con LSODA \parencite{petzoldAutomaticSelectionMethods1983} a traves de \texttt{solve\_ivp} (SciPy), un metodo multipaso de paso y orden variables que conmuta automaticamente entre formulas de Adams (regimen no rigido) y formulas BDF (rigido), con control del error local (tolerancias relativa $10^{-6}$ y absoluta $10^{-9}$). Se prefirio este esquema adaptativo a un metodo de paso fijo tipo Runge--Kutta (RK4): el campo vectorial cambia rapidamente en torno a la transicion de prepupa ($w_p = 0.48$ d) y en los eventos de reposicion, y el control automatico de error garantiza la precision sin ajustar el paso manualmente. La positividad de $D_M$ y $L$ se impuso en el campo vectorial. La implementacion se hizo en Python 3 con NumPy y SciPy.

Los supuestos de la solucion son:
\begin{itemize}
  \item La camara esta bien mezclada y a $T$ y $P$ constantes, de modo que la concentracion medida representa todo el volumen de aire.
  \item Durante un cierre (escala de minutos) las fuentes son constantes frente a la dinamica biologica (escala de dias), por lo que la acumulacion es lineal y la pendiente es la tasa aparente instantanea.
  \item Entre cierres la camara se ventila: el gas no se acumula de un cierre al siguiente y por tanto no se integra como estado.
  \item En los eventos de reposicion los estados larvarios $B$ y $L$ son continuos y solo $D_M$ y $W$ saltan.
  \item La evaporacion del lecho se desprecia por tratarse de una camara cerrada.
\end{itemize}'''
assert s.count(old1) == 1
p.write_text(s.replace(old1, new1))
print("patch_15 aplicado: metodo numerico + supuestos")
