.PHONY: test demo clean

test:
	python3 -m unittest discover -s tests -v

demo:
	python3 -m agent_evals.cli --suite examples/calculator.json --agent examples.calculator_agent:agent
	python3 -m agent_evals.cli --suite examples/kv_store.json --agent examples.kv_agent:agent

clean:
	rm -rf __pycache__ agent_evals/__pycache__ examples/__pycache__ tests/__pycache__ reports
