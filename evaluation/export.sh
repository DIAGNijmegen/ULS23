#!/usr/bin/env bash

./build.sh

docker save uls23 | gzip -c > uls23.tar.gz
