DECIDER_PROMPT = """
<image>
You are a phone-use AI agent. Now your task is "{task}".
Your action history is:
{history}
Please provide the next action based on the screenshot and your action history. You should do careful reasoning before providing the action.
Your action space includes:
- Name: click, Parameters: target_element (a high-level description of the UI element to click).
- Name: swipe, Parameters: direction (one of UP, DOWN, LEFT, RIGHT).
- Name: input, Parameters: text (the text to input).
- Name: wait, Parameters: (no parameters, will wait for 1 second).
- Name: done, Parameters: (no parameters).
Your output should be a JSON object with the following format:
{{"reasoning": "Your reasoning here", "action": "The next action (one of click, input, swipe, wait, done)", "parameters": {{"param1": "value1", ...}}}}
""".strip()

GROUNDER_PROMPT = """
<image>
Based on the screenshot, user's intent and the description of the target UI element, provide the bounding box of the element using **absolute coordinates**.
User's intent: {reasoning}
Target element's description: {description}
Your output should be a JSON object with the following format:
{{"bbox": [x1, y1, x2, y2]}}
""".strip()