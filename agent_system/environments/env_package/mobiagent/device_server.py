import tempfile
import time
import uiautomator2 as u2
import adbutils
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Callable, Any, Optional, Tuple
import io
from abc import ABC, abstractmethod
import uuid
from hmdriver2.driver import Driver
from hmdriver2.hdc import list_devices
import traceback
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os

class CommandRequest(BaseModel):
    command: str
    parameters: Dict[str, Any]

class Device(ABC):
    @abstractmethod
    def start_app(self, app_name):
        raise NotImplementedError

    @abstractmethod
    def screenshot(self):
        raise NotImplementedError

    @abstractmethod
    def click(self, x, y):
        raise NotImplementedError

    @abstractmethod
    def input(self, text):
        raise NotImplementedError

    @abstractmethod
    def swipe(self, direction, scale):
        raise NotImplementedError


class HarmonyDevice(Device):
    def __init__(self, endpoint=None):
        super().__init__()
        self.d = Driver(endpoint)
        self.app_package_names = {
            "携程": "com.ctrip.harmonynext",
            "飞猪": "com.fliggy.hmos",
            "IntelliOS": "ohos.hongmeng.intellios",
            "同城": "com.tongcheng.hmos",
            "携程旅行": "com.ctrip.harmonynext",
            "饿了么": "me.ele.eleme",
            "知乎": "com.zhihu.hmos",
            "哔哩哔哩": "yylx.danmaku.bili",
            "微信": "com.tencent.wechat",
            "小红书": "com.xingin.xhs_hos",
            "QQ音乐": "com.tencent.hm.qqmusic",
            "高德地图": "com.amap.hmapp",
            "淘宝": "com.taobao.taobao4hmos",
            "微博": "com.sina.weibo.stage",
            "京东": "com.jd.hm.mall",
            "飞猪旅行": "com.fliggy.hmos",
            "天气": "com.huawei.hmsapp.totemweather",
            "什么值得买": "com.smzdm.client.hmos",
            "闲鱼": "com.taobao.idlefish4ohos",
            "慧通差旅": "com.smartcom.itravelhm",
            "PowerAgent": "com.example.osagent",
            "航旅纵横": "com.umetrip.hm.app",
            "滴滴出行": "com.sdu.didi.hmos.psnger",
            "电子邮件": "com.huawei.hmos.email",
            "图库": "com.huawei.hmos.photos",
            "日历": "com.huawei.hmos.calendar",
            "心声社区": "com.huawei.it.hmxinsheng",
            "信息": "com.ohos.mms",
            "文件管理": "com.huawei.hmos.files",
            "运动健康": "com.huawei.hmos.health",
            "智慧生活": "com.huawei.hmos.ailife",
            "豆包": "com.larus.nova.hm",
            "WeLink": "com.huawei.it.welink",
            "设置": "com.huawei.hmos.settings",
            "懂车帝": "com.ss.dcar.auto",
            "美团外卖": "com.meituan.takeaway",
            "大众点评": "com.sankuai.dianping",
            "美团": "com.sankuai.hmeituan",
            "浏览器": "com.huawei.hmos.browser",
            "微博": "com.sina.weibo.stage",
            "饿了么": "me.ele.eleme",
            "拼多多": "com.xunmeng.pinduoduo.hos"
        }

    def start_app(self, app_name):
        package_name = self.app_package_names.get(app_name, None)
        if not package_name:
            raise ValueError(f"App '{app_name}' is not registered with a package name.")
        self.d.unlock()
        self.d.force_start_app(package_name)
        time.sleep(2)

    def screenshot(self):
        path = Path(tempfile.gettempdir()) / (uuid.uuid4().hex + ".jpg")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.d.screenshot(str(path))
        with open(path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        # delete the file
        path.unlink(missing_ok=True)
        return img_b64

    def click(self, x, y):
        self.d.click(x, y)
        time.sleep(0.5)

    def input(self, text):
        self.d.shell("uitest uiInput keyEvent 2072 2017")
        self.d.press_key(2071)
        self.d.input_text(text)

    def swipe(self, direction, scale=0.5):
        # self.d.swipe_ext(direction, scale=scale)
        if direction.lower() == "up":
            self.d.swipe(0.5,0.7,0.5,0.3,speed=2000)
        elif direction.lower() == "down":
            self.d.swipe(0.5,0.3,0.5,0.7,speed=2000)
        elif direction.lower() == "left":
            self.d.swipe(0.7,0.5,0.3,0.5,speed=2000)
        elif direction.lower() == "right":
            self.d.swipe(0.3,0.5,0.7,0.5,speed=2000)

class AndroidDevice():
    def __init__(self, endpoint=None):
        self.d = u2.connect(endpoint)

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
            "华为商城": "com.vmall.client",
            "华为视频": "com.huawei.himovie",
            "华为音乐": "com.huawei.music",
            "华为应用市场": "com.huawei.appmarket",
            "拼多多": "com.xunmeng.pinduoduo",
            "大众点评": "com.dianping.v1",
            "小红书": "com.xingin.xhs",
            "浏览器": "com.microsoft.emmx"
        }

    def start_app(self, app_name):
        package_name = self.app_package_names.get(app_name, None)
        if package_name is None:
            raise ValueError(f"App '{app_name}' is not supported.")
        self.d.app_start(package_name, stop=True)
        time.sleep(1)
        if not self.d.app_wait(package_name, timeout=10):
            raise RuntimeError(f"Failed to start package '{package_name}'")
    
    def screenshot(self):
        img = self.d.screenshot()
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return img_b64

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

app = FastAPI()

EXECUTABLE_COMMANDS: list[dict[str, Callable]] = []

devices: list[Device] = []

# 创建线程池，用于异步执行同步的设备操作
executor = ThreadPoolExecutor(max_workers=(os.cpu_count() * 2) if os.cpu_count() is not None else 8)

@app.get("/num_workers/")
async def num_workers():
    return {"num_workers": len(devices)}

@app.post("/{worker_id}/execute_command/")
async def execute_command(worker_id: int, request: CommandRequest):
    """
    接收命令和参数，并执行对应的函数。
    """
    command_name = request.command
    params = request.parameters

    if worker_id >= len(EXECUTABLE_COMMANDS):
        return {"status": "error", "message": f"Worker ID out of range: {worker_id}. Maximum worker ID is {len(EXECUTABLE_COMMANDS) - 1}"}

    if command_name not in EXECUTABLE_COMMANDS[worker_id]:
        return {"status": "error", "message": f"Unknown command: {command_name}"}

    try:
        func_to_execute = EXECUTABLE_COMMANDS[worker_id][command_name]
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(executor, lambda: func_to_execute(**params))

        return {"status": "success", "data": data}
    except TypeError as e:
        traceback.print_exc()
        return {"status": "error", "message": f"Invalid parameters for command '{command_name}': {type(e).__name__}: {e}"}
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": f"Error executing command '{command_name}': {type(e).__name__}: {e}"}

def register_commands():
    global EXECUTABLE_COMMANDS, devices
    for device in devices:
        EXECUTABLE_COMMANDS.append({
            "start_app": device.start_app,
            "click": device.click,
            "input": device.input,
            "swipe": device.swipe,
            "screenshot": device.screenshot
        })

def connect(device_type: str, endpoint: Optional[str] = None):
    global devices
    if endpoint is None:
        if device_type == "android":
            devices.append(AndroidDevice())
        elif device_type == "harmony":
            device_serials = list_devices()
            if len(device_serials) == 0:
                raise RuntimeError("No Harmony devices found")
            else:
                devices.append(HarmonyDevice(device_serials[0]))
    elif endpoint == "all":
        if device_type == "android":
            for device in adbutils.adb.iter_device():
                devices.append(AndroidDevice(device.serial))
        elif device_type == "harmony":
            device_serials = list_devices()
            for serial in device_serials:
                devices.append(HarmonyDevice(serial))
    else:
        if args.device_type == "android":
            device = AndroidDevice(endpoint)
        else:
            device = HarmonyDevice(endpoint)
        devices.append(device)

if __name__ == "__main__":
    import uvicorn, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--device-type", type=str, choices=["android", "harmony"], default="android")
    parser.add_argument("-e", "--device-endpoint", type=str, default=None)
    parser.add_argument("-p", "--port", type=int, default=8000)

    args = parser.parse_args()

    connect(args.device_type, args.device_endpoint)
    register_commands()

    port = args.port
    uvicorn.run(app, host="0.0.0.0", port=port)
