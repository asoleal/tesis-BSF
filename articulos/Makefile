# Compilador y opciones
TEX = xelatex
BIB = biber
FLAGS = -interaction=nonstopmode -halt-on-error

# Carpetas de los artículos
ART1_DIR = articulo_1_sensor_virtual
ART2_DIR = articulo_2_modelo_pinn
ART3_DIR = articulo_3_acv_dinamico
# TESIS_DIR = tesis_completa

.PHONY: all art1 art2 art3 clean

all: art1 art2 art3

# Regla para compilar el Artículo 1
art1:
	@echo "Compilando Artículo 1..."
	cd $(ART1_DIR) && $(TEX) $(FLAGS) main_art1.tex
	cd $(ART1_DIR) && $(BIB) main_art1
	cd $(ART1_DIR) && $(TEX) $(FLAGS) main_art1.tex
	cd $(ART1_DIR) && $(TEX) $(FLAGS) main_art1.tex
	@echo "¡Compilación del Artículo 1 terminada!"

# Regla para compilar el Artículo 2
art2:
	@echo "Compilando Artículo 2..."
	cd $(ART2_DIR) && $(TEX) $(FLAGS) main_art2.tex
	cd $(ART2_DIR) && $(BIB) main_art2
	cd $(ART2_DIR) && $(TEX) $(FLAGS) main_art2.tex
	cd $(ART2_DIR) && $(TEX) $(FLAGS) main_art2.tex
	@echo "¡Compilación del Artículo 2 terminada!"

# Regla para compilar el Artículo 3
art3:
	@echo "Compilando Artículo 3..."
	cd $(ART3_DIR) && $(TEX) $(FLAGS) main_art3.tex
	cd $(ART3_DIR) && $(BIB) main_art3
	cd $(ART3_DIR) && $(TEX) $(FLAGS) main_art3.tex
	cd $(ART3_DIR) && $(TEX) $(FLAGS) main_art3.tex
	@echo "¡Compilación del Artículo 3 terminada!"

# Limpiar archivos auxiliares
clean:
	find . -type f -name "*.aux" -delete
	find . -type f -name "*.bbl" -delete
	find . -type f -name "*.bcf" -delete
	find . -type f -name "*.blg" -delete
	find . -type f -name "*.log" -delete
	find . -type f -name "*.out" -delete
	find . -type f -name "*.run.xml" -delete
	find . -type f -name "*.toc" -delete
