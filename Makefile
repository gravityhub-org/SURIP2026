.PHONY: pull push status
pull:
	uv run dvc pull
push:
	uv run dvc push
status:
	uv run dvc status
