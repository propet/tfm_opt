# Makefile for converting presentation.md to HTML using Marp
# https://marpit.marp.app/

# Variables
INPUT = presentation.md
OUTPUT = presentation.html
MARPCMD = marp

# Default rule
all: $(OUTPUT)

interactive:
	$(MARPCMD) --server --watch .

# Rule to convert markdown to HTML
$(OUTPUT): $(INPUT)
	$(MARPCMD) $(INPUT) --html --output $(OUTPUT)

# Clean rule
clean:
	rm -f $(OUTPUT)

# Phony targets
.PHONY: all clean
