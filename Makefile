check_dirs := distill_bloom

style:
	black --preview $(check_dirs)
	isort $(check_dirs)