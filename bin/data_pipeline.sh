#!/usr/bin/env bash
make check && make data_pipeline ; make clean
dvc status -q || export MESSAGE="./artifacts/data/ has been updated."
printenv MESSAGE && make update_artifacts && unset MESSAGE