import time
import uiautomator2 as u2
import base64
import tempfile, os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Callable, Any, Optional, Tuple
import io
from abc import ABC, abstractmethod
import uuid
from intellicore.kits.appuse.mobiagent.config import supported_apps
from intellicore.kits.deviceuse.devices.openharmony.device import OpenHarmonyDevice
import traceback

class CommandRequest(BaseModel):
    command: str
    parameters: Dict[str, Any]

class Device(ABC):
    @abstractmethod
    def start_app(self, app_name):
        pass

    @abstractmethod
    async def screenshot(self):
        pass

    @abstractmethod
    def click(self, x, y):
        pass

    @abstractmethod
    def input(self, text):
        pass

    @abstractmethod
    def swipe(self, direction, scale):
        pass


class HarmonyDevice:

    def __init__(self, endpoint: str) -> None:
        self.d = OpenHarmonyDevice.getDevice(endpoint)
        self.app_package_names = {app: package for app, (package, _) in supported_apps.items()}
        # Cache screen size; initialize as None and fetch on first use
        self._screen_width: Optional[int] = None
        self._screen_height: Optional[int] = None

    def _get_screen_size(self) -> Tuple[int, int]:
        """Get and cache screen size from a screenshot (only once)."""
        return self._screen_width, self._screen_height

    def start_app(self, app_name: str) -> None:
        package_name = self.app_package_names.get(app_name, None)
        if package_name is None:
            raise ValueError(f"App '{app_name}' is not supported.")
        if app_name == "IntelliOS":
            self.d.startAbilityByUid(package_name, "MainAbility")
        else:
            self.d.restartApplicationByUid(package_name)
        time.sleep(1.0)

    async def screenshot(self) -> str:
        img = await self.d.asyncScreenshot()
        self._screen_width, self._screen_height = img.size
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return img_b64

    def click(self, x: int, y: int) -> None:
        self.d.click((x, y))

    def input(self, text: str) -> None:
        self.d.sendText(text)

    def swipe(self, direction: str, scale: float = 0.5) -> None:
        """
        Scroll in a given direction.

        Args:
            direction (str): One of 'up', 'down', 'left', 'right'.
            scale (float): Proportion of screen size to scroll (0.0 to 1.0).
        """
        width, height = self._get_screen_size()
        center_x, center_y = width / 2.0, height / 2.0

        # Compute offset based on scale
        offset_x = width * scale / 2.0
        offset_y = height * scale / 2.0

        direction = direction.lower()
        if direction == "up":
            start = (center_x, center_y + offset_y)
            end = (center_x, center_y - offset_y)
        elif direction == "down":
            start = (center_x, center_y - offset_y)
            end = (center_x, center_y + offset_y)
        elif direction == "left":
            start = (center_x + offset_x, center_y)
            end = (center_x - offset_x, center_y)
        elif direction == "right":
            start = (center_x - offset_x, center_y)
            end = (center_x + offset_x, center_y)
        else:
            raise ValueError(f"Invalid scroll direction: {direction}. Use 'up', 'down', 'left', or 'right'.")

        self.d.swipe(
            (float(start[0]), float(start[1])),
            (float(end[0]), float(end[1])),
            duration=0.25
        )

class AndroidDevice():
    def __init__(self, endpoint=None):
        self.d = None
        if endpoint:
            self.d = u2.connect(endpoint)
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

    def start_app(self, app_name):
        package_name = self.app_package_names.get(app_name, None)
        if package_name is None:
            raise ValueError(f"App '{app_name}' is not supported.")
        self.d.app_start(package_name, stop=True)
        time.sleep(1)
        if not self.d.app_wait(package_name, timeout=10):
            raise RuntimeError(f"Failed to start package '{package_name}'")
    
    async def screenshot(self):
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

EXECUTABLE_COMMANDS = {}
EXECUTABLE_ASYNC_COMMANDS = {}

device: Optional[Device] = None

@app.post("/execute_command/")
async def execute_command(request: CommandRequest):
    """
    接收命令和参数，并执行对应的函数。
    """
    command_name = request.command
    params = request.parameters

    if (command_name not in EXECUTABLE_COMMANDS) and (command_name not in EXECUTABLE_ASYNC_COMMANDS):
        return {"status": "error", "message": f"Unknown command: {command_name}"}


    try:
        if command_name in EXECUTABLE_COMMANDS:
            func_to_execute = EXECUTABLE_COMMANDS[command_name]
            data = func_to_execute(**params)
        else:
            func_to_execute = EXECUTABLE_ASYNC_COMMANDS[command_name]
            data = await func_to_execute(**params)

        return {"status": "success", "data": data}
    except TypeError as e:
        traceback.print_exc()
        return {"status": "error", "message": f"Invalid parameters for command '{command_name}': {type(e).__name__}: {e}"}
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": f"Error executing command '{command_name}': {type(e).__name__}: {e}"}

def register_commands():
    global EXECUTABLE_COMMANDS, EXECUTABLE_ASYNC_COMMANDS, device
    if device is None:
        raise RuntimeError("Device not initialized. Cannot register commands.")
    EXECUTABLE_COMMANDS = {
        "start_app": device.start_app,
        "click": device.click,
        "input": device.input,
        "swipe": device.swipe,
    }
    EXECUTABLE_ASYNC_COMMANDS = {
        "screenshot": device.screenshot,
    }

if __name__ == "__main__":
    import uvicorn, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--device-type", type=str, choices=["android", "harmony"], default="android")
    parser.add_argument("-e", "--device-endpoint", type=str, default=None)
    parser.add_argument("-p", "--port", type=int, default=8000)
    args = parser.parse_args()

    if args.device_type == "android":
        device = AndroidDevice(args.device_endpoint)
    else:
        device = HarmonyDevice(args.device_endpoint)

    register_commands()

    port = args.port
    uvicorn.run(app, host="0.0.0.0", port=port)



