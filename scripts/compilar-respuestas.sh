#!/usr/bin/env bash
set -euo pipefail

IMAGE="ghcr.io/asoleal/tesis-bsf-base:sha-7c75eab"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

run_in_container() {
  docker run --rm -it \
    --network none \
    --user "$(id -u):$(id -g)" \
    --mount type=bind,src="${ROOT_DIR}",target=/work \
    -w /work \
    "${IMAGE}" \
    bash -lc "$1"
}

cmd="${1:-}"

case "${cmd}" in
  respuestas)
    run_in_container "cd presentacion && latexmk -xelatex -synctex=1 -interaction=nonstopmode presentacion_respuestas_completas.tex"
    ;;
  proceso)
    run_in_container "cd presentacion && latexmk -xelatex -synctex=1 -interaction=nonstopmode presentacion_proceso_bioconversion.tex"
    ;;
  modelodeb)
    run_in_container "cd presentacion && latexmk -xelatex -synctex=1 -interaction=nonstopmode presentacion-modelodeb.tex"
    ;;
  clean)
    run_in_container "cd presentacion && latexmk -C presentacion_respuestas_completas.tex"
    ;;
  clean-proceso)
    run_in_container "cd presentacion && latexmk -C presentacion_proceso_bioconversion.tex"
    ;;
  clean-modelodeb)
    run_in_container "cd presentacion && latexmk -C presentacion-modelodeb.tex"
    ;;
  *)
    echo "Uso: scripts/compilar-respuestas.sh respuestas|proceso|modelodeb|clean|clean-proceso|clean-modelodeb"
    exit 1
    ;;
esac
