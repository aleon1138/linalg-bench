all: bench-cblas bench-linalg

bench-linalg:
	cargo build --release --features real_blackbox
	OMP_NUM_THREADS=1 ./target/release/bench-linalg > out.1
	OMP_NUM_THREADS=1 ./bench-numpy.py > out.2
	OMP_NUM_THREADS=1 ./bench-cblas > out.3
	cat out.[123] > out
	rm out.[123]
	./plot-results.py out

bench-cblas: bench-blas.c
	gcc -DNDEBUG -O2 bench-blas.c -Wall -lopenblas -o bench-cblas
