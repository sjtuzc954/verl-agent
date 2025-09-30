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

from agent_system.environments.prompts import GROUNDER_PROMPT

RESIZE_FACTOR = 0.5  # Resize factor for screenshots to reduce size

class AndroidDevice():
    def __init__(self, adb_endpoint=None):
        super().__init__()
        if adb_endpoint:
            self.d = u2.connect(adb_endpoint)
        else:
            self.d = u2.connect()
        self.app_package_names = {
            "携程": "ctrip.android.view",
            "同城": "com.tongcheng.android",
            "飞猪": "com.taobao.trip",
            "去哪儿": "com.Qunar",
            "华住会": "com.htinns",
            "饿了么": "me.ele",
            "支付宝": "com.eg.android.AlipayGphone",
            "淘宝": "com.taobao.taobao",
            "京东": "com.jingdong.app.mall",
            "美团": "com.sankuai.meituan",
            "滴滴出行": "com.sdu.didi.psnger",
            "微信": "com.tencent.mm",
            "微博": "com.sina.weibo",
            "携程": "ctrip.android.view",
        }

    def start_app(self, app):
        package_name = self.app_package_names.get(app)
        if not package_name:
            raise ValueError(f"App '{app}' is not registered with a package name.")
        self.d.app_start(package_name, stop=True)
        time.sleep(1)
        if not self.d.app_wait(package_name, timeout=10):
            raise RuntimeError(f"Failed to start app '{app}' with package '{package_name}'")
    
    def app_start(self, package_name):
        self.d.app_start(package_name, stop=True)
        time.sleep(1)
        if not self.d.app_wait(package_name, timeout=10):
            raise RuntimeError(f"Failed to start package '{package_name}'")
        
    def screenshot(self, path):
        self.d.screenshot(path)

    def click(self, x, y):
        self.d.click(x, y)

    def input(self, text):
        current_ime = self.d.current_ime()
        self.d.shell(['settings', 'put', 'secure', 'default_input_method', 'com.android.adbkeyboard/.AdbIME'])
        time.sleep(1)
        charsb64 = base64.b64encode(text.encode('utf-8')).decode('utf-8')
        self.d.shell(['am', 'broadcast', '-a', 'ADB_INPUT_B64', '--es', 'msg', charsb64])
        time.sleep(1)
        self.d.shell(['settings', 'put', 'secure', 'default_input_method', current_ime])
        time.sleep(1)

    def swipe(self, direction, scale=0.5):
        # self.d.swipe_ext(direction, scale)
        self.d.swipe_ext(direction=direction, scale=scale)

    def keyevent(self, key):
        self.d.keyevent(key)

    def dump_hierarchy(self):
        return self.d.dump_hierarchy()


class MobiAgentWorker:

    def __init__(self, worker_id: str, grounder_url: str, adb_endpoint: str = None):
        self.worker_id = worker_id
        self.grounder_url = grounder_url
        self.adb_endpoint = adb_endpoint
        self.screenshot_path = f"/tmp/verl-agent-androidenv-screenshot-worker-{self.worker_id}.jpg"
        self.last_obs_base64 = None

        self.grounder_client = OpenAI(api_key="0", base_url=self.grounder_url)

    def _get_obs(self):
        self.device.screenshot(self.screenshot_path)

        # resize the screenshot to reduce the size for processing
        img = Image.open(self.screenshot_path)
        img = img.resize((int(img.width * RESIZE_FACTOR), int(img.height * RESIZE_FACTOR)), Image.Resampling.LANCZOS)
        
        # save last observation as base64 for calling grounder in step()
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        self.last_obs_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return np.array(img)
    
    def _call_grounder(self, reasoning: str, target_element: str):
        grounder_prompt = GROUNDER_PROMPT.format(
            reasoning=reasoning,
            description=target_element,
        )
        grounder_response_str = self.grounder_client.chat.completions.create(
            model="",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.last_obs_base64}"}},
                        {"type": "text", "text": grounder_prompt},
                    ]
                }
            ],
            temperature=0
        ).choices[0].message.content
        grounder_response = json.loads(grounder_response_str)
        x1, y1, x2, y2 = grounder_response["bbox"]
        x, y = (x1 + x2) // 2, (y1 + y2) // 2
        # scale back to original size
        x, y = int(x / RESIZE_FACTOR), int(y / RESIZE_FACTOR)
        return x, y

    def step(self, action: dict[str, Any]):
        device = self.device
        reward = 0.0
        info = {"status": "ok", "won": 0}
        done = False

        try:
            action_type = action["action"]
            parameters = action["parameters"]
            reasoning = action["reasoning"]

            if action_type == "click":
                target_element = parameters["target_element"]
                x, y = self._call_grounder(reasoning, target_element)
                device.click(x, y)
            elif action_type == "input":
                device.input(parameters["text"])
            elif action_type == "swipe":
                device.swipe(parameters["direction"].lower())
            elif action_type == "wait":
                time.sleep(1)
            elif action_type == "done":
                done = True
                info["won"] = 1
            else:
                logging.info(f"Unknown action type, skipping execution: {action_type}")

            time.sleep(2)
            
            obs = self._get_obs()
        except Exception as e:
            reward = -1.0
            done = True
            obs = None
            info = {"status": "error", "error": traceback.format_exc(), "won": 0}

        # TODO: how to assign reward?
        return obs, reward, done, info

    def close(self):
        self.device = None

    def reset(self, task: dict[str, str]):
        self.device = AndroidDevice(adb_endpoint=self.adb_endpoint)
        self.device.app_start(task["package_name"])
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
            adb_endpoints: list[str],
            tasks: list[dict],
            grounder_url: str,
            resources_per_worker: dict,
        ):
        if not ray.is_initialized():
            ray.init()

        self.num_processes = num_envs * group_n
        self.num_envs = num_envs
        self.group_n = group_n
        self.adb_endpoints = adb_endpoints
        print(f"Adb endpoints: {adb_endpoints}")

        if len(adb_endpoints) != self.num_processes:
            raise ValueError(
                f'Number of adb_endpoints ({len(adb_endpoints)}) must match num_envs * group_n ({self.num_processes})',
            )

        # tasks: list of {"task_description": str, "package_name": str}
        self.tasks = tasks
        self.grounder_url = grounder_url

        self.picker = NonRepeatingRandomPicker(np.random.RandomState(seed), self.tasks)

        env_worker = ray.remote(**resources_per_worker)(MobiAgentWorker)
        self.workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(worker_id=str(i), grounder_url=grounder_url, adb_endpoint=adb_endpoints[i])
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
    adb_endpoints: list[str],
    tasks: list[dict],
    grounder_url: str,
    resources_per_worker: dict,
):
    return MobiAgentMultiProcEnvs(
        seed=seed,
        num_envs=env_num,
        group_n=group_n,
        adb_endpoints=adb_endpoints,
        tasks=tasks,
        grounder_url=grounder_url,
        resources_per_worker=resources_per_worker
    )
