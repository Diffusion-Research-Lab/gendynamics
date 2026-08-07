BASH        ?= bash
VENV_DIR    ?= .venv
VENV_BIN    = $(CURDIR)/$(VENV_DIR)/bin
PYTHON_ENV  = PYTHONPATH="$(CURDIR)$${PYTHONPATH:+:$$PYTHONPATH}"

.DEFAULT_GOAL := help

.PHONY: setup vendor check help

setup:
	python -m venv "$(VENV_DIR)"
	"$(VENV_BIN)/python" -m pip install --upgrade pip setuptools wheel
	"$(VENV_BIN)/python" -m pip install -e ".[dev,examples]"

vendor:
	PYTHON="$(VENV_BIN)/python" $(BASH) scripts/fetch.vendor.sh

check:
	$(PYTHON_ENV) "$(VENV_BIN)/flake8" --ignore E501 --exclude gendynamics/_vendor gendynamics examples
	$(PYTHON_ENV) "$(VENV_BIN)/pytest" -q

help:
	@printf "Available targets:\n"
	@printf "  %-10s %s\n" "setup" "Create the development environment"
	@printf "  %-10s %s\n" "vendor" "Fetch optional reference implementations"
	@printf "  %-10s %s\n" "check" "Run lint and tests"
