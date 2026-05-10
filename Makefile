# --- Variables ---
TESIS_MAIN = tesis/tesis.tex
PRES_MAIN = presentacion/presentacion.tex
CANDIDATURA_MAIN = presentacion/presentacion-candidatura.tex

IMAGE = ghcr.io/asoleal/tesis-bsf-base:latest
CONTAINER_CMD = podman run --rm --network=none -v $(shell pwd):/src:z $(IMAGE)

LATEXMK = latexmk -pdfxe -interaction=nonstopmode -file-line-error -halt-on-error
LATEXMK_FORCE = latexmk -g -pdfxe -interaction=nonstopmode -file-line-error -halt-on-error

.PHONY: all tesis presentacion candidatura articulos articulo clean clean-tesis \
        docker-tesis docker-presentacion docker-candidatura docker-all

all: tesis presentacion articulos

tesis: $(TESIS_MAIN)
	@echo "==> Compilando Tesis..."
	(cd tesis && $(LATEXMK_FORCE) tesis.tex)

presentacion: $(PRES_MAIN)
	@echo "==> Compilando Presentación..."
	(cd presentacion && $(LATEXMK) presentacion.tex)

candidatura: $(CANDIDATURA_MAIN)
	@echo "==> Compilando Presentación de Candidatura..."
	(cd presentacion && $(LATEXMK) presentacion-candidatura.tex)

articulos:
	@echo "==> Compilando artículos uno por uno..."
	@find articulos -name "main.tex" | while read -r main_file; do \
	dir=$$(dirname "$$main_file"); \
	echo "--------------------------------------------------"; \
	echo "Procesando: $$dir"; \
	echo "--------------------------------------------------"; \
	(cd "$$dir" && $(LATEXMK) main.tex); \
	done

articulo:
	@if [ -z "$(dir)" ]; then \
	echo "Error: Indica la carpeta. Ejemplo: make articulo dir=articulos/01-nombre"; \
	exit 1; \
	fi
	@FILE=$$(ls $(dir)/*.tex | head -n 1); \
	FILENAME=$$(basename $$FILE .tex); \
	echo "==> Compilando $$FILENAME.tex..."; \
	(cd $(dir) && $(LATEXMK) $$FILENAME.tex)

docker-tesis:
	@echo "==> Compilando Tesis en contenedor..."
	$(CONTAINER_CMD) make tesis

docker-presentacion:
	@echo "==> Compilando Presentación en contenedor..."
	$(CONTAINER_CMD) make presentacion

docker-candidatura:
	@echo "==> Compilando Candidatura en contenedor..."
	$(CONTAINER_CMD) make candidatura

docker-all:
	@echo "==> Compilando todo en contenedor..."
	$(CONTAINER_CMD) make all

clean-tesis:
	@echo "==> Limpiando auxiliares de la tesis..."
	(cd tesis && latexmk -C tesis.tex || true)
	@rm -f tesis/tesis.fdb_latexmk tesis/tesis.fls tesis/tesis.bcf tesis/tesis.run.xml tesis/tesis.bbl tesis/tesis.blg tesis/tesis.xdv

clean:
	@echo "==> Limpiando archivos auxiliares..."
	latexmk -C tesis/tesis.tex || true
	latexmk -C presentacion/presentacion.tex || true
	find articulos -name "*.tex" -execdir latexmk -C \; || true
	find . -type d -name "_minted*" -exec rm -rf {} +
	rm -f *.nav *.snm *.vrb
	latexmk -C presentacion/presentacion-candidatura.tex || true
