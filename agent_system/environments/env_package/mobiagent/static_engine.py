import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from utils import simple_visualize_tracev2


@dataclass
class State:
    '''
        app_state 
    '''
    img_path : str = None# 界面截图路径
    
    map_info : Dict[str, Dict] = field(default_factory=lambda: {
        "click": {},
        "swipe": {},
        "input": {},
        "wait": {}
    })
    
    cluster_class : str = None # 状态簇类别
    is_done = False # 是否为终止状态
    
    def to_dict(self):
        data = asdict(self)
        # 清理数据中的元组键
        return self._clean_data(data)
    
    def _clean_data(self, data):
        """递归清理数据，将元组键转换为字符串"""
        if isinstance(data, dict):
            cleaned = {}
            for key, value in data.items():
                if isinstance(key, tuple):
                    new_key = f"TUPLE:{key}"
                else:
                    new_key = key
                cleaned[new_key] = self._clean_data(value)
            return cleaned
        elif isinstance(data, list):
            return [self._clean_data(item) for item in data]
        else:
            return data
    
    @classmethod
    def from_dict(cls, data):
        # 恢复元组键
        data = cls._restore_data(data)
        return cls(**data)
    
    @classmethod
    def _restore_data(cls, data):
        """递归恢复数据中的元组键"""
        if isinstance(data, dict):
            restored = {}
            for key, value in data.items():
                if isinstance(key, str) and key.startswith("TUPLE:"):
                    try:
                        new_key = eval(key[6:])
                    except:
                        new_key = key
                else:
                    new_key = key
                restored[new_key] = cls._restore_data(value)
            return restored
        elif isinstance(data, list):
            return [cls._restore_data(item) for item in data]
        else:
            return data
    
   
@dataclass
class Action:
    '''
        app action
    '''
    act_type : str
    parameters : dict
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)
    
def union_maps(map1, map2):
    '''
        合并两个状态的map_info字典 map1 -合并给> map2【不覆盖】
    '''
    for act_type in map1.keys():
        if act_type in map2.keys():
            for k,v in map1[act_type].items():
                if k not in map2[act_type].keys():
                    map2[act_type][k] = v
                
    return map2



class TraceLink:
    app_name: str
    task_type: str
    task_description: str
    num_states: int
    states: List[State]  # 状态列表
    actions: List[Action] # 动作列表
    class_data : Dict
    def to_dict(self):
        return {
            "app_name": self.app_name,
            "task_type": self.task_type,
            "task_description": self.task_description,
            "num_states": self.num_states,
            "states": [state.to_dict() for state in self.states],
            #"actions": [action.to_dict() for action in self.actions]
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            app_name=data["app_name"],
            task_type=data["task_type"],
            task_description=data["task_description"],
            num_states=data["num_states"],
            states=[State.from_dict(state) for state in data["states"]],
            actions=[Action.from_dict(action) for action in data["actions"]]
        )
        
def save_states(self, states: List[State]):
    """保存状态列表到JSON文件"""
    data = [state.to_dict() for state in states]
    with open(self.filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_states(self) -> List[State]:
    """从JSON文件加载状态列表"""
    try:
        with open(self.filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [State.from_dict(item) for item in data]
    except FileNotFoundError:
        return []
    
class TraceParser:
    """Trace数据解析器"""
    
    def __init__(self):
        self.trace_link = None
        self.states_map = {}  # img_path -> State
        self.class_data = {}
    
    def parse_trace_directory(self, trace_dir: str) -> TraceLink:
        """
        解析trace目录，包含actions.json、react.json和截图文件
        
        Args:
            trace_dir: trace数据目录路径
            
        Returns:
            Trace对象
        """
        trace_dir = Path(trace_dir)
        
        # 解析actions.json
        actions_file = trace_dir / "actions.json"
        if not actions_file.exists():
            raise FileNotFoundError(f"actions.json not found in {trace_dir}")
        
        actions = self._load_parse_json_file(actions_file)
        
        # 解析截图文件
        screenshots = self._parse_screenshots(trace_dir)
        
        # 构建Trace对象
        trace = self._build_trace_link(states=screenshots, actions=actions)
        
        return trace
    
    def _load_parse_json_file(self, file_path: Path) -> Dict[str, Any]:
        """加载JSON文件"""
        Actions = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                jsonf = json.load(f)
                self.app_name = jsonf.get("app_name", "unknown")
                self.task_type = jsonf.get("task_type", "unknown")
                self.task_description = jsonf.get("task_description", [])
                self.num_states = jsonf.get("action_count", 0)
                actions = jsonf.get("actions", [])
                for action in actions:
                    act_type = action.get("type", "")
                    act_param = {}
                    for k, v in action.items():
                        if k != "type" and v is not None:
                            act_param[k] = v
                    action_obj = Action(act_type=act_type, parameters=act_param) 
                    Actions.append(action_obj)      
                return Actions
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {file_path}: {e}")
    
    def _parse_screenshots(self, trace_dir: Path) -> List[State]:
        """解析截图文件"""
        
        def sort_images_by_order(folder_path):
        
            """按照收集顺序排序图片文件"""
            
            image_files = []
            
            # 收集所有图片文件
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    # 解析文件名格式 a_b.jpg
                    match = re.match(r'^(\d+)_(\d+)\.', filename)
                    if match:
                        category = match.group(1)  # a
                        order = int(match.group(2))  # b
                        image_files.append((order, category, filename))
            
            # 按照 b（收集顺序）排序
            sorted_files = sorted(image_files, key=lambda x: x[0])
            files = []
            #class_data = {}
            for order, category, filename in sorted_files:
                files.append(os.path.join(folder_path,filename))
                
            
            return files,sorted_files
        
        States = []
        
        # 查找所有jpg文件（按数字序排序：1.jpg, 2.jpg, ... 10.jpg）
        jpg_files,sort_files = sort_images_by_order(trace_dir) 
        
        for i, jpg_file in enumerate(jpg_files):
            screenshot = State(img_path=str(jpg_file))
            self.states_map[str(jpg_file)] = screenshot
            States.append(screenshot)
        for order, category, filename in sort_files:
            if category not in self.class_data.keys():
                self.class_data[category] = [self.states_map[os.path.join(trace_dir,filename)]]
            else:
                self.class_data[category].append(self.states_map[os.path.join(trace_dir,filename)]) 
        return States

    def _build_trace_link(self,states,actions):
        self.trace_link = TraceLink()
        self.trace_link.app_name = self.app_name
        self.trace_link.task_type = self.task_type
        self.trace_link.task_description = self.task_description
        self.trace_link.num_states = self.num_states
        self.trace_link.states = states
        self.trace_link.actions = actions
        self.trace_link.class_data = self.class_data
        return self.trace_link
def generate_square_vertices(center_x, center_y, side_length=5):
    """
    生成以给定点为中心的正方形的四个顶点坐标
    
    参数:
        center_x: 中心点的x坐标
        center_y: 中心点的y坐标  
        side_length: 正方形的边长
        
    返回:
        包含四个顶点坐标的列表，顺序为：[左上, 右上, 右下, 左下]
    """
    half_side = side_length 
    
    # 计算四个顶点的坐标
    top_left = (center_x - half_side, center_y + half_side)
    #top_right = (center_x + half_side, center_y + half_side)
    bottom_right = (center_x + half_side, center_y - half_side)
    #bottom_left = (center_x - half_side, center_y - half_side)
    
    return [top_left[0], top_left[1], bottom_right[0], bottom_right[1]]
def add_transition(start_state:State, action:Action, target_state:State):
    
    if action.parameters["isdone"] == True:
        target_state.is_done = True
        
    if action.act_type == "click":
        
        if 'bounds' in action.parameters.keys():
            key = action.parameters['bounds']
            start_state.map_info["click"][tuple(key)] = target_state.img_path
        else:
            key = generate_square_vertices(action.parameters['position_x'], action.parameters['position_y'])
            start_state.map_info["click"][tuple(key)] = target_state.img_path
        
                
                
    elif action.act_type == "swipe":
        dir_ = action.parameters["direction"]
        dis = 0
        
        if dir_ == "up" or dir_ == "down":
            dis = math.fabs(action.parameters["press_position_y"] - action.parameters["release_position_y"])
            
        elif dir_ == "left" or dir_ == "right":
            dis = math.fabs(action.parameters["press_position_x"] - action.parameters["release_position_x"])
            
        start_state.map_info["swipe"][(dir_,dis)] = target_state.img_path
    elif action.act_type == "input":
        start_state.map_info["input"][action.parameters["text"]] = target_state.img_path
    elif action.act_type == "wait":
        start_state.map_info["wait"][action.parameters["duration"]] = target_state.img_path
        
def merge_info(trace:TraceLink):
    '''
        将trace中的action信息合并至state中
    '''
    for i in range(trace.num_states-1):

        add_transition(trace.states[i], trace.actions[i],trace.states[i+1] )
    trace.states[-1].is_done = True # 最后一个状态为终止状态    
def merge_class_info(traces:TraceLink):
    '''
        将多个trace的class_data信息合并
    '''
    
    print("reduce transitions ...")
    for k,v in traces.class_data.items():
        all_map = {
            "click":{
            },
            "swipe":{
            },
            "input":{
            },
            "wait":{
            }
        }
        for s in v:
            for act_type, map_ in s.map_info.items():
                #print(act_type,map_)
                if act_type in all_map.keys():
                    all_map[act_type].update(map_)
        #print(all_map)
        for s in v:
            union_maps(all_map, s.map_info)  

class StateGraph:
    
    def __init__(self,app,task) -> None:
        
        self.app = app
        self.task = task
        self.hash_map = {} #img_path -> state
        self.parser = TraceParser()
        file_dir_os = os.path.dirname(os.path.abspath(__file__))
        self.graph = self.parser.parse_trace_directory(f"{file_dir_os}/static_data/{app}/{task}")
        #self.graph = {}
        self._build_graph()
    def _build_graph(self):
        merge_info(self.graph)
        merge_class_info(self.graph)
        self.hash_map.update(self.parser.states_map)
    def random_init_state(self):
        import random
        return random.choice(self.graph.class_data[0])
    
    def render(self):
        print(self.graph.class_data)
        for state in self.graph.states:
            print(f"State Image Path: {state.img_path}, Map Info: {state.map_info}")
        simple_visualize_tracev2(self.graph)
        
def build_StateGraph(app,task):
    return StateGraph(app,task) 
           
        
    
