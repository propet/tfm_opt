# Makefile for LaTeX project

# Name of the main LaTeX file
MAIN = main

# Compiling command
LATEX = pdflatex -shell-escape
BIBER = biber  # Uncomment if you use BibLaTeX

# List of included files
DEPS = main.tex \
       capitulos/resumen/main.tex \
       capitulos/introduccion/main.tex \
       capitulos/objetivos/main.tex \
       capitulos/metodologia/main.tex \
       capitulos/resultados_discusion/main.tex \
       capitulos/conclusiones/main.tex \
       capitulos/bibliografia/main.tex \
       capitulos/planificacion_presupuesto/main.tex \

# Main build target
all: $(MAIN).pdf

# Compile the main PDF
$(MAIN).pdf: $(DEPS)
	$(LATEX) $(MAIN).tex
	$(LATEX) $(MAIN).tex  # Recompile to resolve references
	# Uncomment the next two lines if you use biblatex/biber for your bibliography
	$(BIBER) $(MAIN)
	$(LATEX) $(MAIN).tex

# Clean up auxiliary LaTeX files
clean:
	rm -f *.aux *.log *.toc *.out *.bbl *.blg *.bcf *.xml *.run.xml *.nav *.snm *.vrb *.fls *.fdb_latexmk

.PHONY: all clean view
