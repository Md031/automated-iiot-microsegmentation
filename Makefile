# Define variables
PYTHON := python3
REQUIREMENTS := requirement.txt
RUN_SCRIPT := scripts/run.py
OPTICS_SCRIPT := scripts/optics.py
DT_SCRIPT := scripts/decision_tree.py

# Default dataset argument (can be overridden in the terminal)
DATASET ?= iotid20

.PHONY: all install start run optics dt clean

# Default target
all: install run

# Install dependencies
install:
	@echo "Installing Python dependencies..."
	$(PYTHON) -m pip install -r $(REQUIREMENTS)

start:
	@echo "Running all the steps for $(DATASET)"
	$(PYTHON) $(RUN_SCRIPT) $(DATASET) && $(PYTHON) $(OPTICS_SCRIPT) $(DATASET) && $(PYTHON) $(DT_SCRIPT) $(DATASET) 

# Run the test/load script
run:
	@echo "Running script for $(DATASET)..."
	$(PYTHON) $(RUN_SCRIPT) $(DATASET)

# Run the optics script
optics:
	@echo "Running optics analysis for $(DATASET)..."
	$(PYTHON) $(OPTICS_SCRIPT) $(DATASET)

# Run the decision tree script
dt:
	@echo "Running decision tree model for $(DATASET)..."
	$(PYTHON) $(DT_SCRIPT) $(DATASET)

# Clean up __pycache__ and temporary files (optional)
clean:
	@echo "Cleaning LaTeX build files..."
	rm -f Report/*.aux \
	      Report/*.bbl \
	      Report/*.blg \
	      Report/*.brf \
	      Report/*.glo \
	      Report/*.gls \
	      Report/*.glg \
	      Report/*.ist \
	      Report/*.acn \
	      Report/*.acr \
	      Report/*.alg \
	      Report/*.sbl \
	      Report/*.slg \
	      Report/*.sym \
	      Report/*.toc \
	      Report/*.lof \
	      Report/*.lot \
	      Report/*.out \
	      Report/*.fdb_latexmk \
	      Report/*.fls \
	      Report/*.synctex.gz \
	      Report/*.log

	@echo "Cleaning Python cache..."
	rm -rf __pycache__ */__pycache__
	rm -f *.pyc *.pyo