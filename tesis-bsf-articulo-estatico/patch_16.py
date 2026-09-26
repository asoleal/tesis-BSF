# patch_16.py - sec. 2.4 analisis de incertidumbre + nota en resultados
from pathlib import Path

p = Path("secciones/metodos.tex"); s = p.read_text()
ancla = "El codigo fuente y los datos quedan disponibles en el repositorio del articulo."
i = s.index(ancla) + len(ancla)
nuevo = r'''

\subsection{Analisis de incertidumbre}
\label{sec:incertidumbre}

Las fuentes de incertidumbre consideradas fueron: (i) la exactitud instrumental del sensor de CO$_2$ (MH-410D, $\pm(50$ ppm $+\ 5$ \% de la lectura) segun el fabricante \parencite{winsenMH410D2022}), que se propaga a la pendiente de cada cierre como un error relativo aproximado de $5$ \% $+\ 50/\Delta C$, con $\Delta C$ el incremento de concentracion durante el cierre (entre 6 y 12 \% para las tasas de este trabajo); (ii) la calidad del ajuste lineal de cada curva, controlada con el filtro $R^2 \geq 0.5$ y el tratamiento de los cierres saturados como cotas inferiores; (iii) la dispersion entre repeticiones, reportada como rango cuando hubo mas de una curva util por punto; y (iv) la incertidumbre del volumen de aire y de las condiciones de la camara (menor al 2 \%). La incertidumbre de la tasa neta se propago como $\sigma_{\mathrm{neto}} = \sqrt{\sigma_{\mathrm{trat}}^2 + \sigma_{\mathrm{ctrl}}^2}$, suponiendo errores independientes entre el tratamiento y su control del mismo dia. El tiempo de respuesta del sensor ($T_{90} < 30$ s) es despreciable frente a la duracion de los cierres. El sensor de CH$_4$ (MH-440D) no estaba calibrado, por lo que sus lecturas se excluyeron de la validacion cuantitativa.'''
p.write_text(s[:i] + nuevo + s[i:])

pr = Path("secciones/resultados.tex"); sr = pr.read_text()
old2 = "La componente total (larvaria mas microbiana) reproduce el orden de magnitud"
new2 = ("Cabe notar que el RMSE de la componente larvaria en D4 (5.0 ppm/min) es del orden de la "
"incertidumbre instrumental estimada para esas tasas (6 a 10 ppm/min, sec. "
"\\ref{sec:incertidumbre}), de modo que la componente larvaria reproduce las tasas netas "
"dentro de la incertidumbre experimental; las discrepancias restantes (la tendencia creciente "
"de los controles, el punto saturado del dia 11 de D1) son estructurales y no instrumentales.\n\n"
+ old2)
assert sr.count(old2) == 1
pr.write_text(sr.replace(old2, new2))
print("patch_16 aplicado: incertidumbre en metodos + nota en resultados")
