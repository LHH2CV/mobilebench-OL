import openai
from openai import OpenAI
import base64
from typing import List, Dict, Any, Optional, Tuple
import re

import pathlib
import mimetypes
sys_prompt = """
You are now operating in Executable Language Grounding mode. Your goal is to help users accomplish tasks by suggesting executable actions that best fit their needs. Your skill set includes both basic and custom actions:

1. Basic Actions
Basic actions are standardized and available across all platforms. They provide essential functionality and are defined with a specific format, ensuring consistency and reliability. 
Basic Action 1: CLICK 
    - purpose: Click at the specified position.
    - format: CLICK <point>[[x-axis, y-axis]]</point>
    - example usage: CLICK <point>[[101, 872]]</point>
Basic Action 2: TYPE
    - purpose: Enter specified text at the designated location.
    - format: TYPE [input text]
    - example usage: TYPE [Shanghai shopping mall]

Basic Action 3: SCROLL
    - purpose: SCROLL in the specified direction.
    - format: SCROLL [direction (UP/DOWN/LEFT/RIGHT)]
    - example usage: SCROLL [UP]
    
2. Custom Actions
Custom actions are unique to each user's platform and environment. They allow for flexibility and adaptability, enabling the model to support new and unseen actions defined by users. These actions extend the functionality of the basic set, making the model more versatile and capable of handling specific tasks.
  
Custom Action 1: PRESS_BACK
    - purpose: Press a back button to navigate to the previous screen.
    - format: PRESS_BACK
    - example usage: PRESS_BACK

Custom Action 2: WAIT
    - purpose: Wait for the screen to load.
    - format: WAIT
    - example usage: WAIT

Custom Action 3: COMPLETE
    - purpose: Indicate the task is finished.
    - format: COMPLETE
    - example usage: COMPLETE

In most cases, task instructions are high-level and abstract. Carefully read the instruction and action history, then perform reasoning to determine the most appropriate next action. Ensure you strictly generate two sections: Thoughts and Actions.
Thoughts: Clearly outline your reasoning process for current step.
Actions: Specify the actual actions you will take based on your reasoning. You should follow action format above when generating. 

Your current task instruction, action history, and associated screenshot are as follows:
Screenshot: 
"""
def to_data_uri(path_or_b64: str) -> str:
    """
    把本地路径或已是 dataURI 的字符串统一转成 dataURI。
    """
    if path_or_b64.startswith("data:image"):
        return path_or_b64                 # 已是 dataURI
    mime = mimetypes.guess_type(path_or_b64)[0] or "image/png"
    data = pathlib.Path(path_or_b64).read_bytes()
    return f"data:{mime};base64," + base64.b64encode(data).decode()
def encode_image(image_path: str) -> str:
    """
    Encodes an image file into a base64 string.

    Parameters:
        image_path (str): The path to the image file.

    Returns:
        str: The base64-encoded string representation of the image.

    Raises:
        FileNotFoundError: If the specified image file is not found.
        IOError: If an error occurs during file reading.
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {image_path}")
    except IOError as e:
        raise IOError(f"Error reading file {image_path}: {e}")
    
class OpenAI_Client:
    def __init__(self, ip, port=8000, api_key="123456"):
        print(f"http://{ip}:{port}/v1")
        openai_api_key = api_key
        openai_api_base = f"http://{ip}:{port}/v1"
        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

        models = self.client.models.list()
        self.model = models.data[0].id
        print(f"opai:{self.model}")

    def call(self, messages, temparature, top_p, max_tokens):
        try:
            if top_p is not None:
                result = self.client.chat.completions.create(
                    messages=messages,
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temparature,
                    top_p=top_p,
                )
            else:
                result = self.client.chat.completions.create(
                    messages=messages,
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temparature
                )
            return result.choices[0].message.content
        except Exception as e:
            print(e)

class os_altas_message_handler(object):
    




    def process_message(
        self,
        task: str,
        image_path: str,
        history: Optional[Dict[str, List[str]]] = None,
    ) -> List[Dict[str, Any]]:

        sys_prompt_block = {
            "role": "user",
            "content": [
                {"type": "text", "text": sys_prompt},
            ],
        }

        # 起始 messages
        messages: List[Dict[str, Any]] = [sys_prompt_block]

        # ------------- 拼接历史 -------------
        if history:
            response_list   = history.get("history_response", [])
            screenshot_list = history.get("history_image_path", [])
        
            # 只保留「回复‑截图」成对数据里的最后 9 条
            pairs = list(zip(response_list, screenshot_list))[-9:]
        
            for reply, shot in pairs:
                # user 上一轮截图 + query
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": to_data_uri(shot)},
                            },
                        ],
                    }
                )
                # assistant 上一轮回答
                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": reply}],
                    }
                )
        # ------------- 当前轮输入 -------------
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": to_data_uri(image_path)},
                    },
                    {
                        "type": "text",
                        "text": f"Task: {task}",
                    },
                ],
            }
        )

        return messages



    def process_response(self, response: str, width: int, height: int) -> Dict:
        def _extract_action_block(resp: str) -> str:
            m = re.search(r"actions:\s*(.+)", resp, flags=re.I | re.S)
            return (m.group(1) if m else '').strip()

        try:
            if not response:
                return {"action": "terminate", "params": {}, "normalized_params": {}}

            s = _extract_action_block(response)

            # ---------- TYPE ----------
            m = re.match(r"TYPE\s*\[(.*?)\]", s, flags=re.I | re.S)
            if m:
                text = m.group(1).strip()
                return {
                    "action": "type",
                    "params": {"text": text},
                    "normalized_params": {"text": text},
                }

            # ---------- CLICK ----------
            m = re.search(r"\[\[\s*(\d+)\s*,\s*(\d+)\s*\]\]", s)
            if s.upper().startswith("CLICK") and m:
                x, y = map(int, m.groups())
                return {
                    "action": "click",
                    "params": {"position": [x * width // 1000, y * height // 1000], "click_times": 1},
                    "normalized_params": {"position": [x / 1000, y / 1000], "click_times": 1},
                }

            # ---------- SCROLL_* → swipe ----------
            m = re.match(r"SCROLL\s*\[(UP|DOWN|LEFT|RIGHT)\]", s, flags=re.I)
            if m:
                direction = m.group(1).upper()
                cx, cy = 0.5, 0.5
                offset = 0.2
                dx = {"LEFT": -offset, "RIGHT": offset}.get(direction, 0)
                dy = {"UP": -offset, "DOWN": offset}.get(direction, 0)
                sx, sy = cx, cy
                ex, ey = cx + dx, cy + dy
                return {
                    "action": "swipe",
                    "params": {
                        "start_position": [int(width * sx), int(height * sy)],
                        "end_position":   [int(width * ex), int(height * ey)],
                        "press_duration": -1,
                    },
                    "normalized_params": {
                        "start_position": [sx, sy],
                        "end_position":   [ex, ey],
                        "press_duration": -1,
                    },
                }

            if re.match(r"PRESS_BACK", s, flags=re.I):
                return {"action": "back", "params": {}, "normalized_params": {}}
            if re.match(r"PRESS_HOME", s, flags=re.I):
                return {"action": "home", "params": {}, "normalized_params": {}}
            if re.match(r"WAIT", s, flags=re.I):
                return {"action": "wait", "params": {}, "normalized_params": {}}
            if re.match(r"(COMPLETE|END\(\))", s, flags=re.I):
                return {"action": "terminate", "params": {}, "normalized_params": {}}
            return {"action": "invalid", "params": {}, "normalized_params": {}}

        except Exception as e:
            raise ValueError(f"[process_response] Failed to parse response: {repr(e)} | Raw: {response}")



class os_altas_Wrapper():


  RETRY_WAITING_SECONDS = 20

  def __init__(
      self,
      max_retry: int = 2,
      temperature: float = 0.0,
      max_length: int = 256,
  ):

    if max_retry <= 0:
      max_retry = 3
      print('Max_retry must be positive. Reset it to 3')
    self.max_retry = min(max_retry, 5)
    self.temperature = temperature
    self.max_length=max_length
    #self.url=url
    self.client=OpenAI_Client("10.221.105.108", port=42309)
    self.message_handler = os_altas_message_handler()
    self.message = []




  def predict_mm(self, goal, current_image_path, history):

    req_messages = self.message_handler.process_message(goal,current_image_path,history)
    response = self.client.call(req_messages, temparature=0, max_tokens=512, top_p=0.9)
    output = self.message_handler.process_response(response,1080,2400)
    return response, output
