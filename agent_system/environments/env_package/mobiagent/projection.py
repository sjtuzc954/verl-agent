import json

def mobiagent_projection(actions: list[str]):
    valids = [1] * len(actions)
    actions = [json.loads(a) for a in actions]
    return actions, valids
    