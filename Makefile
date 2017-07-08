export SHELL := /bin/bash

test:
	pytest --cov pyewa --cov-report term
