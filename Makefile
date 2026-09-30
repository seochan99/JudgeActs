.PHONY: test data dev main analyze paper check

test:
	uv run pytest -q
data:
	uv run python -m src.fetch dataset
	uv run python -m src.prepare
dev:
	uv run python -m src.run_judge --model qwen --split dev --permutations 3
main:
	uv run python -m src.run_judge --model qwen --split main --permutations 3
analyze:
	uv run python -m analysis.analyze --split main
paper:
	uv run python -m analysis.render_paper
check:
	uv run python -m src.validate --split main
	uv run python -m src.check_paper
