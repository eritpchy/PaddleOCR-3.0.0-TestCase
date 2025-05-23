#!/bin/bash
set -e
if [[ $EUID -ne 0 ]]; then exec sudo "$0" "$@"; fi
cd ${0%/*}/
docker build . -t ppocr-test
docker run -it --name ppocr-test --rm --gpus all -v $PWD:/demo --tmpfs /tmp ppocr-test python /demo/test.py 