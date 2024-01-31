#!/usr/bin/env bash

./build.sh

docker save uls23 | gzip -c > ULS23.tar.gz
