import json

def mobiagent_projection(actions: list[str]):
    valids = [1] * len(actions)
    parsed_actions = []
    for i in range(len(actions)):
        try:
            parsed_action = json.loads(actions[i])
            parsed_actions.append(parsed_action)
        except:
            valids[i] = 0
            parsed_actions.append({"reasoning": "", "action": "done", "parameters": {}})

    return parsed_actions, valids
    