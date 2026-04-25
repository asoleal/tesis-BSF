# --- Variables ---
# Nombres de archivos principales
TESIS_MAIN = tesis/tesis.tex
PRES_MAIN  = presentacion/presentacion.tex

# Comando base de latexmk
LATEXMK = latexmk -pdfxe -interaction=nonstopmode -file-line-error -halt-on-error
LATEXMK_FORCE = latexmk -g -pdfxe -interaction=nonstopmode -file-line-error -halt-on-error

# --- Reglas Principales ---

.PHONY: all tesis presentacion articulos clean articulo clean-tesis

# Por defecto, si escribes solo 'make', compila todo
all: tesis presentacion articulos

# Compilar la Tesis
tesis: $(TESIS_MAIN)
	@echo "==> Compilando Tesis..."
	(cd tesis && $(LATEXMK_FORCE) tesis.tex)

# Compilar la Presentación (Beamer)
presentacion: $(PRES_MAIN)
	@echo "==> Compilando Presentación..."
	(cd presentacion && $(LATEXMK) presentacion.tex)

# Compilar todos los artículos encontrados en las subcarpetas
articulos:
	@echo "==> Compilando artículos uno por uno..."
	@find articulos -name "main.tex" | while read -r main_file; do \
		dir=$$(dirname "$$main_file"); \
		echo "--------------------------------------------------"; \
		echo "Procesando: $$dir"; \
		echo "--------------------------------------------------"; \
		(cd "$$dir" && $(LATEXMK) main.tex); \
	done

# --- Regla para compilar un artículo específico ---
articulo:
	@if [ -z "$(dir)" ]; then \
		echo "Error: Indica la carpeta. Ejemplo: make articulo dir=articulos/01-nombre"; \
		exit 1; \
	fi
	@echo "==> Detectando archivo en $(dir)..."
	@FILE=$$(ls $(dir)/*.tex | head -n 1); \
	FILENAME=$$(basename $$FILE .tex); \
	echo "==> Compilando $$FILENAME.tex..."; \
	(cd $(dir) && $(LATEXMK) $$FILENAME.tex)

# Limpiar solo la tesis
clean-tesis:
	@echo "==> Limpiando auxiliares de la tesis..."
	(cd tesis && latexmk -C tesis.tex || true)
	@rm -f tesis/tesis.fdb_latexmk tesis/tesis.fls tesis/tesis.bcf tesis/tesis.run.xml tesis/tesis.bbl tesis/tesis.blg tesis/tesis.xdv

# Limpiar archivos temporales de LaTeX
clean:
	@echo "==> Limpiando archivos auxiliares..."
	latexmk -C tesis/tesis.tex || true
	latexmk -C presentacion/presentacion.tex || true
	find articulos -name "*.tex" -execdir latexmk -C \; || true
	find . -type d -name "_minted*" -exec rm -rf {} +
	rm -f *.nav *.snm *.vrb
	latexmk -C presentacion/presentacion-candidatura.tex || true

CANDIDATURA_MAIN = presentacion/presentacion-candidatura.tex

.PHONY: all tesis presentacion candidatura articulos clean

candidatura: $(CANDIDATURA_MAIN)
	@echo "==> Compilando Presentación de Candidatura..."
	(cd presentacion && $(LATEXMK) presentacion-candidatura.tex)
