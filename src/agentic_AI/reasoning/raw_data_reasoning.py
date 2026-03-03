## reasoning_layer.py
# imports
# from pydantic import BaseModel
# from typing import List
import json

from src.utils.agent_helper import get_openai_client
from src.agentic_AI.agent.prompts import build_llm_prompt


# recommendations = {
#     "skewness": 'recommendation: "log-transform candidate" if abs(skew) > 2',
#     "kurtosis": 'recommendation: "heavy-tailed distribution – outlier robust scaling" if kurtosis > 10'
# }


def run_llm_reasoning(risk_summary, state):
    # # lazy import to avoid circular imports
    # from src.core.agent_classes import LLMAnalysis

    # client = get_openai_client()

    # prompt = build_llm_prompt(risk_summary, state.goal)
    #                         #   dataset_attributes)

    # response = client.chat.completions.create(
    #     model=state.model,
    #     messages=[
    #         {"role": "system", 
    #          "content": "You are a strict JSON-only auditing assistant."},
    #         {"role": "user", 
    #          "content": prompt}
    #     ],
    #     temperature=state.temperature
    # )

    # content = response.choices[0].message.content

    # try:
    #     parsed = json.loads(content)
    #     return LLMAnalysis(**parsed)

    # except Exception:
    #     # Fallback if model breaks JSON format
    #     return LLMAnalysis(
    #         executive_summary="LLM returned unparsable output.",
    #         risk_assessment="Unknown.",
    #         root_cause_hypotheses=[],
    #         recommendations=[],
    #         confidence="low"
    #     )
