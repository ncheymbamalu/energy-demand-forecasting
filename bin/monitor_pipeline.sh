#!/usr/bin/env bash
make check && make monitor_pipeline ; make clean
dvc status -q || export MESSAGE="Either the model artifacts or forecast data have been updated."
printenv MESSAGE && make update_artifacts && unset MESSAGE