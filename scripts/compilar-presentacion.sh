#!/bin/bash
set -e

IMAGE="ghcr.io/asoleal/tesis-bsf-base:sha-7c75eab"

if [ -z "$1" ]; then
    echo "Uso: $0 <archivo.tex>"
    exit 1
fi

FILE="$1"

docker run --rm -it --network none \\
    -v "$(pwd):/work" \\
    -w /work \\
    "$IMAGE" \\
    latexmk -xelatex -synctex=1 -interaction=nonstopmode "$FILE"
