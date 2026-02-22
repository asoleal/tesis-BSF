DOCKER_IMG = tesis-bsf
CURRENT_DIR = $(shell pwd)
UID = $(shell id -u)
GID = $(shell id -g)

DOCKER_CMD = docker run --rm --user $(UID):$(GID) -v "$(CURRENT_DIR):/src"

.PHONY: all tesis slides articulos clean help

help:
	@echo "Opciones: make tesis, make slides, make articulos, make clean"

all: tesis slides articulos

tesis:
	# Compilar tesis: trabajar en /src/tesis
	$(DOCKER_CMD) -w /src/tesis $(DOCKER_IMG) latexmk -xelatex -synctex=1 -interaction=nonstopmode -file-line-error -outdir=. tesis.tex

slides:
	# Compilar presentación: trabajar en /src/presentacion
	$(DOCKER_CMD) -w /src/presentacion $(DOCKER_IMG) latexmk -xelatex -f -synctex=1 -interaction=nonstopmode -file-line-error -outdir=. presentacion.tex

articulos:
	# Buscar y compilar todos los main.tex en articulos/
	@for art in articulos/*/main.tex; do \
		if [ -f "$$art" ]; then \
			dir=$$(dirname $$art); \
			echo "Compilando $$art en $$dir"; \
			$(DOCKER_CMD) -w /src/$$dir $(DOCKER_IMG) latexmk -xelatex -f -synctex=1 -interaction=nonstopmode -file-line-error -outdir=. main.tex; \
		fi \
	done

clean:
	# Limpiar archivos temporales en tesis/
	cd tesis && rm -f *.aux *.log *.out *.toc *.lof *.lot *.bbl *.blg *.fls *.fdb_latexmk *.synctex.gz *.bcf *.run.xml *.xdv
	cd tesis && rm -f Capitulos/*.aux contenido/*.aux preambulo/*.aux 2>/dev/null || true
	# Limpiar en presentacion/
	cd presentacion && rm -f *.aux *.log *.out *.toc *.nav *.snm *.vrb *.fls *.fdb_latexmk *.synctex.gz *.bcf *.run.xml *.xdv 2>/dev/null || true
	# Limpiar en articulos/
	find articulos -name "*.aux" -o -name "*.log" -o -name "*.out" -o -name "*.toc" -o -name "*.fls" -o -name "*.fdb_latexmk" -o -name "*.synctex.gz" -o -name "*.bcf" -o -name "*.run.xml" -o -name "*.xdv" | xargs rm -f 2>/dev/null || true
	@echo "Limpieza completada."
