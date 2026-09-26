# patch_4.py - mejora estetica de la figura TikZ del nucleo larvario
from pathlib import Path
p = Path("secciones/metodos.tex"); s = p.read_text()
i = s.index("\\begin{tikzpicture}")
j = s.index("\\end{tikzpicture}") + len("\\end{tikzpicture}")
NUEVO = r'''\begin{tikzpicture}[
  comp/.style={draw, rounded corners=2mm, minimum width=3.0cm, minimum height=1.1cm, align=center, fill=gray!8},
  gas/.style={draw, dashed, rounded corners=2mm, minimum width=2.2cm, minimum height=1.1cm, align=center},
  flujo/.style={->, >=latex, thick, shorten >=3pt, shorten <=3pt},
  etiqueta/.style={fill=white, inner sep=1.5pt, font=\footnotesize}]
\node[comp] (alim) at (0,0) {Lecho\\ (alimento)};
\node[comp] (A) at (5.4,0) {$A$: asimilado};
\node[comp] (B) at (10.8,0) {$B$: estructura};
\node[comp] (L) at (10.8,-3.8) {$L$: lipidos};
\node[gas] (CO2) at (5.4,-3.8) {CO$_2$};
\draw[flujo] (alim) -- node[etiqueta, above=2pt]{$r_A = a\,B\,(1 - S)$}
                      node[etiqueta, below=2pt, font=\scriptsize\itshape]{interruptor de prepupa $S(t)$} (A);
\draw[flujo] (A) -- node[etiqueta, above=2pt]{$(1 + Y_B)\,r_B$} (B);
\draw[flujo] (A) -- node[etiqueta, right=2pt]{$r_{Cm} + Y_B\,r_B$} (CO2);
\draw[flujo] (A.south east) -- node[etiqueta, sloped, above=2pt]{$r_L$} (L.north west);
\draw[flujo] (L) -- node[etiqueta, below=2pt]{$Y_L\,r_L$} (CO2);
\end{tikzpicture}'''
p.write_text(s[:i] + NUEVO + s[j:])
print("patch_4 aplicado: figura mejorada")
