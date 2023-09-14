## Makefile
#
# Development scripts for BrainLM huggingface repo
#
# @author Syed Rizvi <syed.rizvi@yale.edu>
#

.PHONY: test

test:
	pytest tests
