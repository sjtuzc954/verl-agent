import uiautomator2 as u2
import time
import base64
import logging
from PIL import Image
import io
import ray
import numpy as np
from typing import Any
from openai import OpenAI
import json
import traceback
import numpy as np
import requests
import os
from functools import wraps

from agent_system.environments.prompts import GROUNDER_PROMPT, REWARD_PROMPT, IDENTIFY_FAIL_ACTION_PROMPT, FOLLOW_UP_IDENTIFY_STEP_PROMPT

RESIZE_FACTOR = 0.5  # Resize factor for screenshots to reduce size

def retry(max_attempts=3, default=None, check_fn=None):
    """
    Decorator that retries a function up to max_attempts times
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    result = func(*args, **kwargs)
                    if check_fn is not None and not check_fn(result):
                        raise ValueError(f"Result does not satisfy check_fn: {result}")
                    return result
                except Exception as e:
                    print(f"{func.__name__} failed with error: {e.__class__.__name__}: {e}")

                if attempt < max_attempts:
                    print(f"retrying... (attempt {attempt}/{max_attempts})")
                
            print(f"{func.__name__} failed after {max_attempts} attempts")
            return default
        return wrapper
    return decorator

class InvalidActionError(RuntimeError):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

class MobiAgentWorker:

    def __init__(
        self, 
        worker_id: str, 
        grounder_url: str = None, 
        device_server_url: str = None,
        use_rel_coords: bool = False, 
        use_e2e: bool = False,
        reward_kwargs: dict = {}
    ):
        self.worker_id = worker_id
        self.grounder_url = grounder_url
        self.device_server_url = device_server_url
        self.use_rel_coords = use_rel_coords
        self.use_e2e = use_e2e

        self.screenshot_path = f"verl-agent-androidenv-screenshot-worker-{self.worker_id}.jpg"
        self.last_obs_base64 = None
        if self.grounder_url is not None and (not self.use_e2e):
            self.grounder_client = OpenAI(api_key="0", base_url=self.grounder_url)
        else:
            self.grounder_client = None
        
        os.environ["http_proxy"] = "http://127.0.0.1:7897"
        os.environ["https_proxy"] = "http://127.0.0.1:7897"
        os.environ["no_proxy"] = "localhost,127.0.0.1"
        self.reward_model = reward_kwargs.pop("model")
        self.reward_mode = reward_kwargs.pop("mode")
        self.reward_client = OpenAI(**reward_kwargs)

        self.is_done = False
        self.traj = []
        self.current_task = None
        self.last_obs = None
        
    def _get_obs(self):
        response = requests.post(f"{self.device_server_url}/execute_command/", json={
            "command": "screenshot",
            "parameters": {}
        })

        if response.status_code != 200:
            raise RuntimeError("Failed to get screenshot from device server")
        response_body = response.json()
        if response_body.get("status") != "success":
            raise RuntimeError(f"Device server returned error: {response_body.get('message', 'Unknown error')}")

        img_base64 = response_body.get("data")
        
        img_bytes = base64.b64decode(img_base64)
        img = Image.open(io.BytesIO(img_bytes))
        img = img.resize((int(img.width * RESIZE_FACTOR), int(img.height * RESIZE_FACTOR)), Image.Resampling.LANCZOS)
        
        self.last_obs = img

        return np.array(img)
    
    def _call_grounder(self, reasoning: str, target_element: str):
        grounder_prompt = GROUNDER_PROMPT.format(
            reasoning=reasoning,
            description=target_element,
        )
        buffer = io.BytesIO()
        self.last_obs.save(buffer, format="JPEG")
        last_obs_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        grounder_response_str = self.grounder_client.chat.completions.create(
            model="",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{last_obs_base64}"}},
                        {"type": "text", "text": grounder_prompt},
                    ]
                }
            ],
            temperature=0
        ).choices[0].message.content
        grounder_response = json.loads(grounder_response_str)
        x1, y1, x2, y2 = grounder_response["bbox"]
        x, y = (x1 + x2) // 2, (y1 + y2) // 2
        if self.use_rel_coords:
            x = int(x / 1000 * self.last_obs.width)
            y = int(y / 1000 * self.last_obs.height)
        # scale back to original size
        x, y = int(x / RESIZE_FACTOR), int(y / RESIZE_FACTOR)
        return x, y

    def step(self, action: str):
        # device = self.device
        reward = 0.0
        info = {"status": "ok", "won": 0}
        done = False
        
        if self.is_done:
            obs = np.array(self.last_obs) if self.last_obs else None
            return obs, reward, True, info

        step_idx = len(self.traj)
        try:
            print(f"Worker {self.worker_id}, step {step_idx}, raw action: {action}")
            
            action = json.loads(action)

            action_type = action["action"]
            if action_type not in ["click", "input", "swipe", "wait", "done"]:
                raise InvalidActionError(f"Unknown action type: {action_type}")

            parameters = action["parameters"]
            reasoning = action["reasoning"]

            self.traj.append([self.last_obs, action])

            request_body = None

            if action_type == "click":
                target_element = parameters["target_element"]
                if self.grounder_client is not None and (not self.use_e2e):
                    x, y = self._call_grounder(reasoning, target_element)
                elif self.use_e2e and "bbox" in parameters:
                    bbox = parameters["bbox"]
                    x, y = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
                    if self.use_rel_coords:
                        x = int(x / 1000 * self.last_obs.width / RESIZE_FACTOR)
                        y = int(y / 1000 * self.last_obs.height / RESIZE_FACTOR)
                else:
                    raise InvalidActionError(f"Invalid click action: {action}")
                request_body = {
                    "command": "click",
                    "parameters": {"x": x, "y": y}
                }
            elif action_type == "input":
                # device.input(parameters["text"])
                request_body = {
                    "command": "input",
                    "parameters": {"text": parameters["text"]}
                }
            elif action_type == "swipe":
                # device.swipe(parameters["direction"].lower())
                request_body = {
                    "command": "swipe",
                    "parameters": {"direction": parameters["direction"].lower()}
                }
            elif action_type == "wait":
                # time.sleep(1)
                pass
            elif action_type == "done":
                done = True
                self.is_done = True
                reward, failed_step = self._get_reward()
                # if parameters.get("status", "success") == "success":
                #     # _get_reward now handles both reward evaluation and failed step identification
                #     reward, failed_step = self._get_reward()
                # else:
                #     reward = 0.0
                #     failed_step = None
                
                info["won"] = reward == 1.0
                
                # If the task failed and we identified the failed step
                if failed_step is not None:
                    # Convert to 0-indexed for trajectory access
                    info["failed_step_idx"] = failed_step - 1
                    print(f"Identified failed step: {failed_step}, {self.traj[failed_step - 1][1]}")

            if request_body is not None:
                requests.post(f"{self.device_server_url}/execute_command/", json=request_body)

            time.sleep(1.5)

            obs = self._get_obs()
        except Exception as e:
            reward = 0.0
            done = True
            obs = np.array(self.last_obs) if self.last_obs else None
            failed_step = step_idx if isinstance(e, (json.decoder.JSONDecodeError, InvalidActionError, KeyError)) else None
            info = {"status": "error", "error": traceback.format_exc(), "won": 0, "failed_step_idx": failed_step}
            self.is_done = True

        return obs, reward, done, info

    def _build_trajectory_message(self, init_prompt: str, max_images_per_turn: int = 16) -> list[dict]:
        """
        Build trajectory messages for LLM evaluation with image limit per turn.
        
        Args:
            init_prompt: The initial reward prompt text
            max_images_per_turn: Maximum number of images allowed per conversation turn (default: 16)
            
        Returns:
            A list of messages (with role and content) that can span multiple turns if needed.
        """
        messages = []
        current_content = [{"type": "text", "text": init_prompt}]
        current_image_count = 0
        
        # Add trajectory: each step contains [screenshot: PIL.Image, action: dict]
        texts = []
        total_images = len(self.traj)
        
        for i, (screenshot, action) in enumerate(self.traj):
            # Check if we need to start a new turn
            if current_image_count >= max_images_per_turn:
                # Add continuation prompt
                current_content.append({
                    "type": "text",
                    "text": "I haven't finished sending all the screenshots and actions due to API limit. Please reply 'Received' to continue."
                })
                
                # Add current user message
                messages.append({
                    "role": "user",
                    "content": current_content
                })
                
                # Add assistant acknowledgment
                messages.append({
                    "role": "assistant",
                    "content": "Received."
                })
                
                # Start new content for next turn
                current_content = []
                current_image_count = 0
            
            # Add screenshot
            buffer = io.BytesIO()
            screenshot.save(buffer, format="JPEG")
            screenshot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            current_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{screenshot_base64}"}
            })
            current_image_count += 1
            
            # Add action description as text
            action_text = f"Step {i+1} - Action: {action['action']}"
            if action['parameters']:
                action_text += f", Parameters: {action['parameters']}"
            action_text += f", Reasoning: {action['reasoning']}"
            
            current_content.append({
                "type": "text",
                "text": action_text
            })

            texts.append(action_text)

        print(f"Trajectory texts: \n{'\n'.join(texts)}")
        print(f"Total images: {total_images}, split into {len(messages)//2 + 1} turn(s)")
        
        # Add the final user message
        messages.append({
            "role": "user",
            "content": current_content
        })
        
        return messages

    @retry(max_attempts=3, default=(0.0, None), check_fn=lambda x: isinstance(x, tuple) and len(x) == 2 and x[0] in [0.0, 1.0] and (x[1] is None or isinstance(x[1], int)))
    def _get_reward(self):
        """
        Use LLM as judge to evaluate if the task was successfully completed.
        If failed, continues the conversation to identify the failed step.
        
        Returns (reward, failed_step) where:
        - reward: 1.0 if successful (Choice A), 0.0 if not successful (Choice B)
        - failed_step: 1-indexed step number that caused failure, or None if task succeeded
        """
        # Build the reward prompt with task description
        reward_prompt = REWARD_PROMPT.format(task=self.current_task)
        
        # Build trajectory messages (may span multiple turns if many images)
        messages = self._build_trajectory_message(reward_prompt)
        
        # Send messages to evaluate task success
        response = self.reward_client.chat.completions.create(
            model=self.reward_model,
            messages=messages
        )
        
        # Parse the response
        reward_output = response.choices[0].message.content
        print(f"Reward output:\n{reward_output}")
        
        # Parse the output to extract the choice
        # Expected format:
        # Thought: ...
        # Choice: A or B
        reward_output_clean = reward_output.replace("```", "")
        lines = reward_output_clean.strip().split('\n')
        choice = None
        for line in lines:
            if line.startswith("Choice:"):
                choice_str = line.split(":", 1)[1].strip()
                choice = choice_str.strip().upper()
                break
        
        if choice == "A":
            return 1.0, None
        elif choice == "B":
            if self.reward_mode != "process":
                return 0.0, None
            # Second round: identify failed step using multi-turn conversation
            messages.append({
                "role": "assistant",
                "content": reward_output
            })
            messages.append({
                "role": "user",
                "content": FOLLOW_UP_IDENTIFY_STEP_PROMPT
            })
            
            response = self.reward_client.chat.completions.create(
                model=self.reward_model,
                messages=messages
            )
            
            fail_output = response.choices[0].message.content
            print(f"Fail step identification output:\n{fail_output}")
            
            # Parse the output to extract the failed step
            # Expected format:
            # Thought: ...
            # Failed Step: [number]
            lines = fail_output.replace("```", "").strip().split('\n')
            failed_step = None
            for line in lines:
                if line.startswith("Failed Step:"):
                    step_str = line.split(":", 1)[1].strip()
                    failed_step = int(step_str)
                    break
            
            if failed_step is None or failed_step < 1 or failed_step > len(self.traj):
                raise ValueError(f"Invalid failed step: {failed_step}, Trajectory length: {len(self.traj)}")
            
            return 0.0, failed_step
        else:
            raise ValueError(f"Invalid choice: {choice}, Reward output: {reward_output}")

    def close(self):
        pass

    def reset(self, task: dict[str, str]):
        # self.device = AndroidDevice(adb_endpoint=self.adb_endpoint)
        # self.device.app_start(task["package_name"])
        self.is_done = False
        self.traj = []
        self.last_obs = None

        response = requests.post(f"{self.device_server_url}/execute_command/", json={
            "command": "start_app",
            "parameters": {"app_name": task["app_name"]}
        })

        print(f"Picked task: {task}")

        self.current_task = task["description"]

        return self._get_obs(), {"task": task["description"]}
    
class NonRepeatingRandomPicker:

    def __init__(self, rng: np.random.RandomState, items):
        self.remaining_items = list(items)
        rng.shuffle(self.remaining_items)

        self.original_items = list(self.remaining_items)

    def pick(self, n) -> list:
        if n > len(self.original_items):
            raise ValueError(f"Cannot pick {n} items from a list of {len(self.original_items)} unique items")
        
        num_to_pick = min(n, len(self.remaining_items))

        picked_items = [self.remaining_items.pop() for _ in range(num_to_pick)]

        if num_to_pick < n:
            self.remaining_items = list(self.original_items)
            picked_items += self.pick(n - num_to_pick)
        
        return picked_items

class MobiAgentMultiProcEnvs:

    def __init__(
            self,
            seed: int,
            num_envs: int,
            group_n: int,
            resources_per_worker: dict,
            device_server_urls: list[str],
            tasks: list[dict],
            grounder_url: str,
            use_rel_coords: bool,
            use_e2e: bool = False,
            reward_kwargs: dict = {}
        ):
        if not ray.is_initialized():
            ray.init()

        self.num_processes = num_envs * group_n
        self.num_envs = num_envs
        self.group_n = group_n
        self.device_server_urls = device_server_urls
        print(f"Device Server URLs: {device_server_urls}")

        if len(device_server_urls) != self.num_processes:
            raise ValueError(
                f'Number of adb_endpoints ({len(device_server_urls)}) must match num_envs * group_n ({self.num_processes})',
            )

        # tasks: list of {"task_description": str, "package_name": str}
        self.tasks = tasks
        self.grounder_url = grounder_url
        self.use_rel_coords = use_rel_coords
        self.use_e2e = use_e2e
        self.picker = NonRepeatingRandomPicker(np.random.RandomState(seed), self.tasks)

        env_worker = ray.remote(**resources_per_worker)(MobiAgentWorker)
        self.workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(
                worker_id=str(i), 
                grounder_url=grounder_url, 
                device_server_url=device_server_urls[i], 
                use_rel_coords=self.use_rel_coords,
                use_e2e=self.use_e2e,
                reward_kwargs=reward_kwargs
            )
            self.workers.append(worker)

    def step(self, actions: list[str]):
        if len(actions) != self.num_processes:
            raise ValueError(
                f'Expected {self.num_processes} actions, got {len(actions)}',
            )
        
        futures = []
        for worker, action in zip(self.workers, actions):
            future = worker.step.remote(action)
            futures.append(future)

        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list
    
    def reset(self):
        random_tasks = self.picker.pick(self.num_envs)
        random_tasks = np.repeat(random_tasks, self.group_n).tolist()

        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.reset.remote(random_tasks[i])
            futures.append(future)

        results = ray.get(futures)
        obs_list, info_list = [], []
        for i, (obs, info) in enumerate(results):
            obs_list.append(obs)
            info_list.append(info)

        return obs_list, info_list
    

    def close(self):
        """Close all workers."""
        # Send close commands to all workers
        futures = []
        for worker in self.workers:
            future = worker.close.remote()
            futures.append(future)
        
        # Wait for all workers to close
        ray.get(futures)
        
        # Shutdown Ray actors
        for worker in self.workers:
            ray.kill(worker)

    def render(self):
        pass


def build_mobiagent_envs(
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: dict,
    env_kwargs: dict,
):
    device_server_urls = env_kwargs["device_server_urls"]
    tasks = env_kwargs["tasks"]
    grounder_url = env_kwargs.get("grounder_url", None)
    use_rel_coords = env_kwargs.get("use_rel_coords", False)
    use_e2e = env_kwargs.get("use_e2e", False)
    reward_kwargs = {
        "model": env_kwargs.get("reward_model", None),
        "api_key": env_kwargs.get("reward_api_key", None),
        "base_url": env_kwargs.get("reward_base_url", None),
        "mode": env_kwargs.get("reward_mode", "episode")
    }
    return MobiAgentMultiProcEnvs(
        seed=seed,
        num_envs=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        device_server_urls=device_server_urls,
        tasks=tasks,
        grounder_url=grounder_url,
        use_rel_coords=use_rel_coords,
        use_e2e=use_e2e,
        reward_kwargs=reward_kwargs
    )
