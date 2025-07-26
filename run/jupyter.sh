#!/bin/bash

notebook_dir="$(dirname "$(dirname "$(realpath "$0")")")"

jupyter lab \
	--ip 0.0.0.0 \
	--port 8888 \
	--allow-root \
	--notebook-dir $notebook_dir