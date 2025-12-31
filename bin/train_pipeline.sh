#!/usr/bin/env bash
make check && make train ; make clean
dvc status -q || export MESSAGE="./artifacts/model/ has been updated."
printenv MESSAGE && make update_model_artifacts && unset MESSAGE