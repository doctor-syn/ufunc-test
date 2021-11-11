
install:
	python setup.py install

clean:
	python setup.py clean

test: src/ufunc.c
	python setup.py build
	rm -rf npufunc_directory/ || true
	mv ./build/lib.linux-x86_64-3.8/npufunc_directory .
	python test.py
