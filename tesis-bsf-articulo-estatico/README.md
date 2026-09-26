# Articulo: validacion estatica del modelo DEB (BSF)

Validacion experimental del modelo DEB de Eriksen para predecir
emisiones de CO2 y CH4 en bioconversion estatica (batch tipo panera)
con *Hermetia illucens*. Repo independiente de tesis-BSF.

## Compilar

    xelatex main; biber main; xelatex main; xelatex main

## Estructura

- `main.tex` — maestro (xelatex + biblatex/biber, sin natbib)
- `secciones/` — resumen, introduccion, metodos, resultados, discusion, conclusiones
- `figuras/` — PDFs generados por `simulacion/`
- `bibliografia/` — export de Zotero
