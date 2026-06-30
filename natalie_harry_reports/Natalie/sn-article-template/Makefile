all: sn-article.pdf

sn-article.pdf: sn-article.tex sn-bibliography.bib sn-mathphys-ay.bst sn-jnl.cls 
	latexmk -pdf sn-article.tex
	echo "created sn-article.pdf"

clean: 
	latexmk -C
	rm sn-article.bbl
