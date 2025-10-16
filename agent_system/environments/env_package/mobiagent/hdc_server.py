from intellicore.kits.screenuse.base import ScreenOperator
from intellicore.kits.deviceuse.devices import Device, AndroidAdbDevice, OpenHarmonyDevice
from intellicore.kits.screenuse.parsers import android_uitree_parser, oh_uitree_parser
from intellicore.kits.screenuse.transforms.icondetect import IconMatchTransform, IconDetectTransform, IconDetectTransformV2
from intellicore.kits.screenuse.transforms.ocr import OcrTransform
from intellicore.kits.appuse.tllm import TextLLMAppOperator
from intellicore.utils.config import SystemConfig
from intellicore.kits.screenuse.semui import SemUIView


import time
import uiautomator2 as u2
import base64
import tempfile, os

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Dict, Any, Callable, Optional
from contextlib import asynccontextmanager




import time
from typing import Tuple
from PIL import Image

from intellicore.kits.appuse.mobiagent.config import supported_apps
from intellicore.kits.deviceuse.devices.openharmony.device import OpenHarmonyDevice


class MobiAgentDevice:

    def __init__(self, device: OpenHarmonyDevice, factor: int) -> None:
        self.d = device
        self.factor = factor
        self.app_package_names = {app: package for app, (package, _) in supported_apps.items()}
        # Cache screen size; initialize as None and fetch on first use
        self._screen_width: Optional[int] = None
        self._screen_height: Optional[int] = None

    def _get_screen_size(self) -> Tuple[int, int]:
        """Get and cache screen size from a screenshot (only once)."""
        if self._screen_width is None or self._screen_height is None:
            img = self.d.screenshot()
            self._screen_width, self._screen_height = img.size
        return self._screen_width, self._screen_height

    def get_apps(self) -> list[str]:
        return list(self.app_package_names.keys())

    def start_app(self, package_name: str) -> None:
        app = next((app for app, package in self.app_package_names.items() if package == package_name), None)
        if app is None:
            raise ValueError(f"Package '{package_name}' is not registered.")
        if app == "IntelliOS":
            self.d.startAbilityByUid(package_name, "MainAbility")
        else:
            self.d.restartApplicationByUid(package_name)
        time.sleep(1.0)

    def screenshot(self) -> Image.Image:
        return self.d.screenshot()

    def click(self, x: int, y: int) -> None:
        self.d.click((x, y))

    def input(self, text: str) -> None:
        self.d.sendText(text)

    def scroll(self, direction: str, scale: float = 0.5) -> None:
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

    def wait(self, time_: float) -> None:
        time.sleep(time_)

# 全局变量，用于存储 device
app_device: Optional['MobiAgentDevice'] = None
EXECUTABLE_COMMANDS: Dict[str, Callable[..., Any]] = {}


class CommandRequest(BaseModel):
    command: str
    parameters: Dict[str, Any]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global app_device, EXECUTABLE_COMMANDS
    try:
        # 尝试获取设备（你可以根据需要改成指定 IP:Port）
        # oh_dev = OpenHarmonyDevice.getFirstDevice()
        oh_dev = OpenHarmonyDevice.getDevice("127.0.0.1:6000")
        if oh_dev is None:
            raise Exception("No OpenHarmony device found")

        app_device = MobiAgentDevice(oh_dev, 1)

        # 注册可执行命令
        EXECUTABLE_COMMANDS = {
            "start_app": app_device.start_app,
            "screenshot": app_device.screenshot,
            "click": app_device.click,
            "input": app_device.input,
            "swipe": app_device.scroll,
        }

        print("Device initialized successfully.")
        yield  # 启动应用
    except Exception as ex:
        print(f"Failed to initialize device: {ex}")
        # 可选择退出或允许 API 启动但返回错误
        yield
    finally:
        # 可选：清理资源（如断开设备）
        if app_device:
            # 如果 MobiAgentDevice 有 close/disconnect 方法，可在这里调用
            pass


app = FastAPI(lifespan=lifespan)


@app.post("/execute_command/")
async def execute_command(request: CommandRequest):
    global app_device, EXECUTABLE_COMMANDS

    if app_device is None:
        raise HTTPException(status_code=503, detail="Device not initialized. Check server logs.")

    command_name = request.command
    params = request.parameters

    if command_name not in EXECUTABLE_COMMANDS:
        raise HTTPException(status_code=404, detail=f"Command '{command_name}' not found.")

    func_to_execute = EXECUTABLE_COMMANDS[command_name]

    try:
        result = func_to_execute(**params)
        return {"status": "success", "result": result}
    except TypeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid parameters for command '{command_name}': {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing command '{command_name}': {e}")



def test_mobi_agent_device():
    print("🔍 正在查找 OpenHarmony 设备...")
    # oh_dev = OpenHarmonyDevice.getFirstDevice()
    oh_dev = OpenHarmonyDevice.getDevice("127.0.0.1:6000")

    if oh_dev is None:
        raise RuntimeError("❌ 未找到 OpenHarmony 设备，请确保设备已连接并开启调试。")

    print("✅ 设备已连接，初始化 MobiAgentDevice...")
    device = MobiAgentDevice(oh_dev, factor=1)

    # 1. 测试 get_apps
    apps = device.get_apps()
    print(f"📦 支持的应用列表: {apps}")
    if not apps:
        print("⚠️  警告：supported_apps 为空，部分测试将跳过。")

    # 2. 测试 start_app（选第一个应用）
    if apps:
        first_app = apps[0]
        package = device.app_package_names[first_app]
        print(f"🚀 启动应用: {first_app} ({package})")
        try:
            device.start_app(package)
            print("✅ 应用启动成功")
        except Exception as e:
            print(f"⚠️  启动应用失败（可能正常）: {e}")

    # 3. 测试 screenshot
    print("📸 获取截图...")
    img = device.screenshot()
    print(f"✅ 截图尺寸: {img.size}")
    img.save("test_screenshot.png")
    print("💾 截图已保存为 test_screenshot.png")

    # 4. 测试 click（点击屏幕中心）
    width, height = img.size
    center_x, center_y = width // 2, height // 2
    print(f"🖱️  点击屏幕中心 ({center_x}, {center_y})")
    device.click(center_x, center_y)
    print("✅ 点击完成")

    time.sleep(1)  # 给 UI 响应时间

    # 5. 测试 input（可选：如果当前界面有输入框）
    print("⌨️  尝试输入文本 'Hello MobiAgent'")
    try:
        device.input("Hello MobiAgent")
        print("✅ 文本输入完成")
    except Exception as e:
        print(f"⚠️  输入失败（可能无焦点）: {e}")
    
    time.sleep(1)  # 给 UI 响应时间

    # 6. 测试 scroll（四个方向）
    print("🔄 测试滚动功能...")
    for direction in ["up", "down", "left", "right"]:
        print(f"  ➡ 滚动方向: {direction}")
        device.scroll(direction, scale=0.4)
        time.sleep(0.5)  # 给 UI 响应时间
    print("✅ 滚动测试完成")

    # 7. 测试 wait
    print("⏳ 等待 1 秒...")
    device.wait(1.0)
    print("✅ 等待完成")

    print("\n🎉 所有测试步骤执行完毕！请人工检查设备行为是否符合预期。")

# 如果你仍想保留 __main__ 用于调试启动
if __name__ == "__main__":
    # test_mobi_agent_device()

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)