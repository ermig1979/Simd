SIMD_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../../)
run:
	cd $(SIMD_DIR) && docker build -t simd -f prj/docker/Dockerfile .
	docker run --name foo -it simd bash -c "mkdir /tmp/simd; mv /simd/build/libSimd.* /tmp/simd"
	docker cp foo:/tmp/simd/. .
	docker rm foo