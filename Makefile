all: PLOTS

PLOTS: build/r0.pdf \
	build/r1.pdf \
	build/dl0.pdf \
	build/dl1a.pdf \
	build/dl1a_clean.pdf


build/r1.pdf build/dl0.pdf build/dl1a.pdf build/dl1a_clean.pdf: build/r0.pdf

build/r0.pdf: matplotlibrc_pgf plot_datalevels.py | build
	MATPLOTLIBRC=matplotlibrc_pgf TEXINPUTS=$$(pwd): python plot_datalevels.py


build:
	mkdir -p $@


FORCE:

clean:
	rm -rf build

.PHONY: all clean FORCE plots preview
