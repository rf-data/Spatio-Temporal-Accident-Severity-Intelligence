## agent_planning.py
# imports
import json

from src.agentic_AI.agent.prompts import build_planning_prompt, build_reflection_prompt

#   build_eda_planning_prompt,
#   build_eda_reflection_prompt)

from src.utils.agent_helper import get_openai_client


def plan_checks(goal, check_dict, model, temp, agent):
    client = get_openai_client()

    if agent == "audit":
        prompt = build_planning_prompt(goal, check_dict, scope="auditing")
    elif agent == "raw_data":
        prompt = build_planning_prompt(goal, check_dict, scope="exploration")
    else:
        raise ValueError(
            f"Input from 'agent' is not valid ({agent}). Prompt cannot be refined."
        )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a strict JSON-only planning agent."},
            {"role": "user", "content": prompt},
        ],
        temperature=temp,
    )

    content = response.choices[0].message.content

    try:
        selected_checks = json.loads(content)
        # Sicherheitsfilter:
        return [c for c in selected_checks if c in check_dict]

    except Exception:
        # Fallback: führe alle Checks aus
        return list(check_dict.keys())


def reflect_and_plan(goal, check_dict, state, agent):
    # setup AI client
    client = get_openai_client()

    # extract values
    executed_checks = state.executed_checks
    findings = state.findings
    model = state.model
    temp = state.temperature

    #
    remaining_checks = {
        name: fn for name, fn in check_dict.items() if name not in executed_checks
    }

    if not remaining_checks:
        return []

    if agent == "audit":
        prompt = build_reflection_prompt(
            goal,
            "data auditing",
            [f.model_dump() for f in findings],
            executed_checks,
            remaining_checks,
        )

    elif agent == "raw_data":
        prompt = build_reflection_prompt(
            goal,
            "data exploration",
            [f.model_dump() for f in findings],
            executed_checks,
            remaining_checks,
        )

    else:
        raise ValueError(f"Input from 'agent' is not valid:\t{agent}")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a strict JSON-only planning agent."},
            {"role": "user", "content": prompt},
        ],
        temperature=temp,
    )

    try:
        selected = json.loads(response.choices[0].message.content)
        return [c for c in selected if c in remaining_checks]

    except Exception:
        return []
