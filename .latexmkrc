# Forzar siempre la compilación a PDF usando XeLaTeX
$pdf_mode = 5;

# Configurar los argumentos de XeLaTeX para que coincidan con tu Makefile
$xelatex = 'xelatex -synctex=1 -interaction=nonstopmode -file-line-error -halt-on-error %O %S';
