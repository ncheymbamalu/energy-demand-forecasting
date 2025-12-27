#!/usr/bin/env bash
make check && make features ; make clean
dvc status -q || export MESSAGE="./artifacts/data/processed.parquet has been updated."
printenv MESSAGE && make update_features && unset MESSAGE