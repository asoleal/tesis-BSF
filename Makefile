# --- Variables ---
# Nombres de archivos principales
TESIS_MAIN = tesis.tex
PRES_MAIN  = presentacion.tex
# Comando base de latexmk
LATEXMK = latexmk -pdfxe -interaction=nonstopmode -file-line-error -halt-on-error

# --- Reglas Principales ---

.PHONY: all tesis presentacion articulos clean

# Por defecto, si escribes solo 'make', compila todo
all: tesis presentacion articulos

# Compilar la Tesis
tesis: $(TESIS_MAIN)
>	@echo "==> Compilando Tesis..."
>	$(LATEXMK) $(TESIS_MAIN)

# Compilar la Presentación (Beamer)
presentacion: $(PRES_MAIN)
>	@echo "==> Compilando Presentación..."
>	$(LATEXMK) $(PRES_MAIN)

# Compilar todos los artículos encontrados en las subcarpetas
articulos:
>	@echo "==> Compilando artículos uno por uno..."
>	@find articulos -name "main.tex" | while read -r main_file; do \
>		dir=$$(dirname "$$main_file"); \
>		echo "--------------------------------------------------"; \
>		echo "Procesando: $$dir"; \
>		echo "--------------------------------------------------"; \
>		(cd "$$dir" && $(LATEXMK) main.tex); \
>	done

# Limpiar archivos temporales de LaTeX
clean:
>	@echo "==> Limpiando archivos auxiliares..."
>	$(LATEXMK) -C
>	find articulos -name "main.tex" -execdir latexmk -C \;
>	rm -rf _minted* *.nav *.snm *.vrb