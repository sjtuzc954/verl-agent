import time
import uiautomator2 as u2
import base64
import tempfile, os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Callable, Any

app = FastAPI()

class CommandRequest(BaseModel):
    command: str
    parameters: Dict[str, Any]

EXECUTABLE_COMMANDS: Dict[str, Callable[..., Any]] = {}

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
    
    def screenshot(self):
        tmpdir = tempfile.gettempdir()
        path = os.path.join(tmpdir, "adb_server_screenshot.jpg")
        self.d.screenshot(path)
        # read image and return base64
        with open(path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode('utf-8')
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

device: AndroidDevice = None

EXECUTABLE_COMMANDS = {
    "start_app": device.start_app,
    "screenshot": device.screenshot,
    "click": device.click,
    "input": device.input,
    "swipe": device.swipe,
}

@app.post("/execute_command/")
async def execute_command(request: CommandRequest):
    """
    接收命令和参数，并执行对应的函数。
    """
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

if __name__ == "__main__":
    import uvicorn, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--adb-endpoint", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    port = args.port
    adb_endpoint = args.adb_endpoint
    device = AndroidDevice(adb_endpoint)
    uvicorn.run(app, host="0.0.0.0", port=port)



