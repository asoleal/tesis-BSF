# --- Variables ---
DOCKER_IMG = tesis-bsf
# Usamos CURDIR que es la variable nativa de Make para la ruta absoluta
CURRENT_DIR = $(CURDIR)

# Comando base: Monta el directorio actual en /src del contenedor
DOCKER_CMD = docker run --rm -v "$(CURRENT_DIR):/src" -w /src $(DOCKER_IMG)

# --- Objetivos ---
.PHONY: all tesis slides clean help

help:
	@echo "Herramientas de compilación de Tesis:"
	@echo "  make tesis    -> Compila Tesis completa (PDFLaTeX + BibTeX + índices)"
	@echo "  make slides   -> Compila Presentación (XeLaTeX)"
	@echo "  make clean    -> Borra todos los archivos temporales"

all: tesis slides

# Compilación completa: Latex -> Bibtex -> Latex -> Latex (para referencias cruzadas)
tesis:
	$(DOCKER_CMD) sh -c "pdflatex 0000.tex && bibtex 0000 && pdflatex 0000.tex && pdflatex 0000.tex"

slides:
	$(DOCKER_CMD) xelatex presentacion.tex

# Limpieza profunda (incluye carpeta contenido)
clean:
	rm -f *.aux *.log *.out *.toc *.lof *.lot *.bbl *.blg *.fls *.fdb_latexmk *.synctex.gz *.nav *.snm *.vrb
	rm -f contenido/*.aux
