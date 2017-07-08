export SHELL := /bin/bash

test:
	py.test --cov=pyewa --cov-report term testing/
