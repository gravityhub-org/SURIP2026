all: sn-article.pdf

sn-article.pdf: sn-article.tex sn-bibliography.bib sn-mathphys-num.bst sn-jnl.cls 
	latexmk -pdf sn-article.tex
	echo "Created sn-article.pdf"


clean: 
	latexmk -C
	rm sn-article.bbl






