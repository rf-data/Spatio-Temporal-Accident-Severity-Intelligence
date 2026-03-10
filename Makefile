ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PROJECT := road_accidents
NAME ?= raw_data_charac

.PHONY: postgre_start audit_agent_run sql_upload raw_data_agent_run

postgre_start:
	psql -h localhost -U road_user -d $(PROJECT)

audit_agent_run:
	python -m src.agentic_AI.run_audit_agent

raw_data_agent_run:
	python -m src.agentic_AI.run_raw_data_agent --name $(NAME)

sql_upload:
	python -m src.agentic_AI.run_sql_upload --name $(NAME)

