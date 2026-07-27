.PHONY: test smoke native image

test:
	. .venv/bin/activate && pytest

smoke:
	. .venv/bin/activate && aeolus-train --config configs/smoke.yaml
	. .venv/bin/activate && aeolus-eval --config configs/smoke.yaml --checkpoint runs/smoke_v4/checkpoint.pt --episodes 8

native:
	c++ -std=c++20 -O3 -Wall -Wextra -pedantic native/src/reference_kernel.cpp native/tests/smoke.cpp -I native/include -o native/aeolus_native_smoke
	./native/aeolus_native_smoke

image:
	docker build -f deploy/Dockerfile -t aeolus-ia:dev .
