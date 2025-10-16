import base64
import io
import json
import logging
import time
import traceback
from typing import Any

import numpy as np
import ray
import requests
import uiautomator2 as u2
#from environments.prompts import GROUNDER_PROMPT
from openai import OpenAI
from PIL import Image
from static_engine import build_StateGraph
from utils import *

#from agent_system.environments.prompts import GROUNDER_PROMPT
GROUNDER_PROMPT = """
Based on the screenshot, user's intent and the description of the target UI element, provide the bounding box of the element using **absolute coordinates**.
User's intent: {reasoning}
Target element's description: {description}
Your output should be a JSON object with the following format:
{{"bbox": [x1, y1, x2, y2]}}"""
RESIZE_FACTOR = 0.5  # Resize factor for screenshots to reduce size

class StaticMobiAgentWorker:
    
    def __init__(self,graph,grounder_url:str, cur_state = None) -> None:
        self.graph = graph
        self.grounder_url = grounder_url
        if cur_state is None:
            self.cur_state = self.graph.random_init_state()
        else:
            self.cur_state = cur_state
        self.last_state = None    
        
    def _get_obs(self):
        img = Image.open(self.cur_state.img_path)
        img = img.resize((int(img.width * RESIZE_FACTOR), int(img.height * RESIZE_FACTOR)), Image.Resampling.LANCZOS)
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
        # scale back to original size
        x, y = int(x / RESIZE_FACTOR), int(y / RESIZE_FACTOR)
        return x, y
    def _cal_reward(self):
        # TODO
        return 0 
    def _handle_click(self,x,y):
        
        for k,v in self.cur_state.map_info["click"].items():
            if point_in_rectangle(x,y,k[0],k[1],k[2],k[3]):
                self.last_state = self.cur_state
                self.cur_state = self.graph.hash_map[v]                
                return
            
    def _handle_swipe(self,direction):
        for k,v in self.cur_state.map_info["swipe"].items():
            dir_ = direction.lower()
            if k[0] == dir_:    
                self.last_state = self.cur_state
                self.cur_state =  self.hash_map[v]
                return 
        
    def _handle_input(self,text):   
        for k,v in self.cur_state.map_info["input"].items():
            if k == text: # match text TODO
                self.last_state = self.cur_state
                self.cur_state =  self.hash_map[v]
                return 
    
    def step(self,action):
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
                self._handle_click(x, y)
                reward = self._cal_reward()
                if self.cur_state.is_done:
                    done = True
                    info["won"] = 1
                    
            elif action_type == "input":
                text = parameters["text"]
                self._handle_input(text)
                reward = self._cal_reward()
                if self.cur_state.is_done:
                    done = True
                    info["won"] = 1
                
            elif action_type == "swipe":
                # device.swipe(parameters["direction"].lower())
                direction = parameters["direction"]
                self._handle_swipe(direction)
                reward = self._cal_reward()
                if self.cur_state.is_done:
                    done = True
                    info["won"] = 1
                    
            elif action_type == "wait":
                time.sleep(1)
                
            elif action_type == "done":
                done = True
                info["won"] = 1
            else:
                logging.info(f"Unknown action type, skipping execution: {action_type}")  
            obs = self._get_obs()
            
        except Exception as e:
            reward = -1.0
            done = True
            obs = None
            info = {"status": "error", "error": traceback.format_exc(), "won": 0}

        # TODO: how to assign reward?
        return obs, reward, done, info       
        
    def reset(self, task: dict[str, str]):
        self.cur_state = self.graph.random_init_state()
        self.last_state = None
        return self._get_obs(), {"task": task["description"]}
         
    def close(self):
        pass
    
class StaticAgentEnvs:
    def __init__(
            self,
            seed: int,
            num_envs: int,
            group_n: int,
            tasks: list[str],
            apps: list[str],
            grouder_url: str,
            resources_per_worker: dict,
        ):
        self.num_workers = num_envs * group_n
        self.num_envs = num_envs
        self.group_n = group_n
        
        self.envs = [build_StateGraph(app=apps[i],task=tasks[i]) for i in range(self.num_envs)]
        self.workers = []
        for i in range(self.num_envs):
            for j in range(self.group_n):
                worker = StaticMobiAgentWorker(self.envs[i],grouder_url)
                self.workers.append(worker)
        
    def step(self,actions):
        if len(actions) != self.num_processes:
            raise ValueError(
                f'Expected {self.num_processes} actions, got {len(actions)}',
            )
        futures = []
        for worker, action in zip(self.workers, actions):
            future = worker.step(action)
            futures.append(future)

        #results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in futures:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list
    
    def reset(self):
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.reset()
            futures.append(future)        
        obs_list, info_list = [], []
        for i, (obs, info) in enumerate(futures):
            obs_list.append(obs)
            info_list.append(info)
        return obs_list, info_list
    
    def close(self):
        pass
    def render(self):
        pass 
    
def build_static_mobiagent_envs(
    seed: int,
    env_num: int,
    group_n: int,
    #adb_urls: list[str],
    tasks: list[dict],
    grounder_url: str,
    resources_per_worker: dict,
):return StaticAgentEnvs(
    seed=seed,
    num_envs=env_num,
    group_n=group_n,
    #adb_urls=adb_urls,
    tasks=tasks,
    grounder_url=grounder_url,
    resources_per_worker=resources_per_worker
)
#adb url
#tasks 
if __name__ == "__main__":
    worker = build_StateGraph("meituan","type2")
    worker.render()