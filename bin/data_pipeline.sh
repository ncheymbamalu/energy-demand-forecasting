#!/usr/bin/env bash
make check && make data ; make clean
dvc status -q || export MESSAGE="./artifacts/data/ has been updated."
printenv MESSAGE && make update_data_artifacts && unset MESSAGE