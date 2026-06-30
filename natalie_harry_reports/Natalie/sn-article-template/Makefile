all: main.pdf

main.pdf: main.tex references.bib sn-mathphys-ay.bst sn-jnl.cls 
	latexmk -pdf main.tex
	echo "created main.pdf"

clean: 
	latexmk -C
	rm main.bbl
