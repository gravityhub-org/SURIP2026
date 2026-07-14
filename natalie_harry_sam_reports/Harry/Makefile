all: main.pdf

main.pdf: main.tex references.bib sn-mathphys-num.bst sn-jnl.cls 
	latexmk -pdf main.tex
	echo "Created main.pdf"


clean: 
	latexmk -C
	rm -f main.bbl






