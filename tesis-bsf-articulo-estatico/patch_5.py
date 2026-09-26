# patch_5.py - tercera version de la figura TikZ: etiquetas cortas + compuerta S
from pathlib import Path
p = Path("secciones/metodos.tex"); s = p.read_text()
i = s.index("\\begin{tikzpicture}")
j = s.index("\\end{tikzpicture}") + len("\\end{tikzpicture}")
NUEVO = r'''\begin{tikzpicture}[
  comp/.style={draw, rounded corners=2mm, minimum width=3.0cm, minimum height=1.1cm, align=center, fill=gray!8},
  gas/.style={draw, dashed, rounded corners=2mm, minimum width=2.0cm, minimum height=1.1cm, align=center},
  gate/.style={circle, draw, thick, fill=white, inner sep=1.8pt, font=\footnotesize},
  flujo/.style={->, >=latex, thick, shorten >=3pt, shorten <=3pt},
  etiqueta/.style={fill=white, inner sep=1.5pt, font=\footnotesize}]
\node[comp] (alim) at (0,0) {Lecho\\ (alimento)};
\node[comp] (A) at (6.0,0) {$A$: asimilado};
\node[comp] (B) at (11.6,0) {$B$: estructura};
\node[comp] (L) at (11.6,-4.0) {$L$: lipidos};
\node[gas] (CO2) at (6.0,-4.0) {CO$_2$};
\draw[flujo] (alim) -- node[etiqueta, above=2pt]{$r_A$}
                      node[gate, below=2pt]{$S$} (A);
\draw[flujo] (A) -- node[etiqueta, above=2pt]{$(1 + Y_B)\,r_B$} (B);
\draw[flujo] (A) -- node[etiqueta, right=2pt]{$r_{Cm} + Y_B\,r_B$} (CO2);
\draw[flujo] (A.south east) -- node[etiqueta]{$r_L$} (L.north west);
\draw[flujo] (L) -- node[etiqueta, below=2pt]{$Y_L\,r_L$} (CO2);
\end{tikzpicture}'''
p.write_text(s[:i] + NUEVO + s[j:])
print("patch_5 aplicado: figura v3")
