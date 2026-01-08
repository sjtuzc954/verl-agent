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
Based on the screenshot, user's intent and the description of the target UI element, provide the bounding box of the element using **absolute coordinates**.
User's intent: {reasoning}
Target element's description: {description}
Your output should be a JSON object with the following format:
{{"bbox": [x1, y1, x2, y2]}}"""

E2E_PROMPT = """
<image>
You are a phone-use AI agent. Now your task is "{task}".
Your action history is:
{history}
Please provide the next action based on the screenshot and your action history. You should do careful reasoning before providing the action.
Your action space includes:
- Name: click, Parameters: target_element (a high-level description of the UI element to click), bbox (an bounding box of the target element,[x1, y1, x2, y2]).
- Name: swipe, Parameters: direction (one of UP, DOWN, LEFT, RIGHT), start_coords (the starting absolute coordinate [x, y]), end_coords (the ending absolute coordinate [x, y]).
- Name: input, Parameters: text (the text to input).
- Name: wait, Parameters: (no parameters, will wait for 1 second).
- Name: done, Parameters: status (the completion status of the current task, one of `success', `suspended` and `failed`).
Your output should be a JSON object with the following format:
{{"reasoning": "Your reasoning here", "action": "The next action (one of click, input, swipe, wait, done)", "parameters": {{"param1": "value1","param2": "value2", ...}}}}
""".strip()

REWARD_PROMPT = """
You are an expert evaluator assessing whether a mobile phone agent has successfully completed a given task.

## Agent's Task Description
{task}

## Agent's Action Trajectory
You will be provided with the agent's complete execution trajectory as a sequence of screenshots and corresponding actions. Each screenshot shows the phone's state, followed by the action taken by the agent at that step.

## Evaluation Criteria
1. Goal Achievement: Assess whether the user's original objective has been fully accomplished. Partial progress does not constitute success.
2. Evidence-Based Assessment: Base your judgment strictly on the visual evidence from screenshots and the action sequence provided.
3. Final State Verification: Pay close attention to the final screenshot to verify if the desired end state was reached.
4. Agent Completion Signals: If the agent called the `done` action with status `success`, consider this as a strong indicator, but verify it against the visual evidence.
5. External Constraints: Distinguish between agent errors and external system constraints that are beyond the agent's control. If the agent has successfully completed all required operational steps but cannot proceed due to external constraints (e.g., minimum order requirements not met, items out of stock, merchant-imposed restrictions, service time limitations), this should still be considered a SUCCESS as long as the agent's action sequence was correct and appropriate for the given task.
6. Trial and Error Tolerance: The agent is allowed to make mistakes and self-correct during the execution process. Focus on the final state and whether the task objective was accomplished, rather than penalizing exploratory or corrective behaviors. 

## Your Task
Carefully examine the entire trajectory—both the screenshots and actions—to determine whether the task was successfully completed.

## Output Format
You must provide your response according to the following **two-line** format (DO NOT include ``` in your response):

```
Thought: [Your detailed reasoning process, analyzing the trajectory and explaining why you reached your conclusion]
Choice: [A or B]
```

Where:
- A means the task was successfully completed and all objectives were achieved.
- B means the task was not completed, either due to failure, partial completion, or suspension.
""".strip()

IDENTIFY_FAIL_ACTION_PROMPT = """
You are an expert evaluator analyzing why a mobile phone agent failed to complete a given task.

## Agent's Task Description
{task}

## Human Judge's Assessment
A human expert evaluator has already reviewed this trajectory and determined that the task was NOT successfully completed. Their reasoning for the failure is provided below:

Failure reason from human judge: {failure_reason}

This assessment should guide your analysis in identifying the specific action that led to this failure. Use the human judge's reasoning as a high-level understanding of what went wrong, then pinpoint the exact step in the trajectory that caused or contributed to this failure.

## Agent's Failed Trajectory
You will be provided with the agent's complete execution trajectory that did NOT successfully complete the task. The trajectory consists of a sequence of screenshots and corresponding actions. Each screenshot shows the phone's state, followed by the action taken by the agent at that step.

## Your Task
Identify the SPECIFIC action (step number) that caused or led to the task failure. Analyze the trajectory carefully to pinpoint where things went wrong and why that particular action was problematic.

## Analysis Guidelines
- Consider these failure patterns:
  - Clicking on the wrong UI element
  - Entering incorrect text or data
  - Navigating to an irrelevant screen
  - Taking an action that contradicts the task goal
  - Missing a critical step in the task flow
  - Getting stuck in a loop or dead-end state
  - Marking the task status as `success` before all objectives were completed (in such case, the failed step is the last `done` action)
- Base your analysis strictly on the visual evidence from screenshots and the action sequence.
- If multiple actions contributed to failure, identify the FIRST action that initiated the failure chain.

## Output Format
You must provide your response according to the following **two-line** format (DO NOT include ``` in your response):

```
Thought: [Your detailed step-by-step analysis explaining why you identified this specific step as the failure point. Describe what you observe in the screenshots and why this action was incorrect.]
Failed Step: [The step number that caused the failure]
```

Where the step number should be an integer corresponding to the step in the trajectory (e.g., if Step 3 caused the failure, output "Failed Step: 3").
""".strip()

FOLLOW_UP_IDENTIFY_STEP_PROMPT = """
Based on your previous assessment that the task failed, please now identify the SPECIFIC action (step number) that caused or led to the task failure.

Analyze the trajectory carefully to pinpoint where things went wrong and why that particular action was problematic.

## Analysis Guidelines
- Consider these failure patterns:
  - Clicking on the wrong UI element
  - Entering incorrect text or data
  - Navigating to an irrelevant screen
  - Taking an action that contradicts the task goal
  - Missing a critical step in the task flow
  - Getting stuck in a loop or dead-end state
  - Marking the task status as `success` before all objectives were completed (in such case, the failed step is the last `done` action)
- Base your analysis strictly on the visual evidence from screenshots and the action sequence you just reviewed.
- If multiple actions contributed to failure, identify the FIRST action that initiated the failure chain.

## Output Format
You must provide your response according to the following **two-line** format (DO NOT include ``` in your response):

```
Thought: [Your detailed step-by-step analysis explaining why you identified this specific step as the failure point. Describe what you observe in the screenshots and why this action was incorrect.]
Failed Step: [The step number that caused the failure]
```

Where the step number should be an integer corresponding to the step in the trajectory (e.g., if Step 3 caused the failure, output "Failed Step: 3").
""".strip()