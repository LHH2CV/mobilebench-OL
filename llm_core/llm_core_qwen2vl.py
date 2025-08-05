from openai import OpenAI
from typing import List, Dict, Any, Optional, Tuple ,Union
import re
import base64, math, requests
from PIL import Image
from io import BytesIO
from utils import representation_utils 
from utils import m3a_utils
from utils import action_parser_tool
from utils import xml_screen_parser_tool
import numpy as np

sys_prompt = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.
## Screenshot size
(0,0) -> (1080,2400)
## Output Format
```
Thought: ...
Action: ...
```

## Action Space
click(start_box='(x1,y1)')
type(content='')
scroll(start_box='(x1,y1)', end_box='(x2,y2)')
press_home()
press_back()
wait()
finished(content='') # Submit the task regardless of whether it succeeds or fails.

## Note
- Use Chinese in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
"""

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

class qwen2vl_message_handler(object):
    
    def process_message(
        self,
        task: str,
        image_path: str,
        history: Optional[Dict[str, List[str]]] = None,
    ) -> List[Dict[str, Any]]:

        sys_prompt_block = {
            "role": "user",
            "content": [
                {"type": "text", "text": sys_prompt + task},
            ],
        }

        # 起始 messages
        messages: List[Dict[str, Any]] = [sys_prompt_block]
        messages = [{
                "role": "system",
                "content": "You are a helpful assistant."
        }] + messages
        # ------------- 拼接历史 -------------
        if history:
            response_list   = history.get("history_response", [])
            screenshot_list = history.get("history_image_path", [])
        
            # 只保留「回复‑截图」成对数据里的最后 9 条
            pairs = list(zip(response_list, screenshot_list))[-5:]
        
            for reply, shot in pairs:
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": action_parser_tool.image_to_uri(shot)},
                            },
                        ],
                    }
                )
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
                        "image_url": {"url": action_parser_tool.image_to_uri(image_path)},
                    }
                ],
            }
        )

        return messages



    def process_response(self, content, width, height):
        result = action_parser_tool.parse_agent_output(content)
        extracted_action = result["action"]
        try:
            if "click" in extracted_action:
                x,y = action_parser_tool.extract_xy_from_point(extracted_action)
                normalized_params = {
                    "position":[x/1000, y/1000],
                    "click_times":1
                }
                params = {
                    "position":[int(x * width / 1000), int(y * height / 1000)],
                    "click_times":1
                }
                result.update({
                    "action": "click",
                    "params": params,
                    "normalized_params": normalized_params
                })
            elif "type" in extracted_action:
                text = re.search(r"content='(.*?)'", extracted_action).group(1)
                result.update({
                    "action": "type",
                    "params": {"text": text},
                    "normalized_params": {"text": text}
                })
            elif "scroll" in extracted_action or "swipe "in extracted_action:
                x1,y1,x2,y2 = action_parser_tool.extract_swipe_points(extracted_action)
                normalized_params = {
                    "start_position": [x1/1000, y1/1000],
                    "end_position": [x2/1000, y2/1000],
                    "press_duration": -1
                }
                params = {
                    "start_position": [int(width * x1 / 1000), int(height * y1/1000)],
                    "end_position": [int(width * x2 / 1000), int(height * y2/1000)],
                    "press_duration": -1
                }
                result.update({
                    "action": "swipe",
                    "params": params,
                    "normalized_params": normalized_params
                })
            elif "navigate_back" in extracted_action or "press_back" in extracted_action:
                result.update({
                    "action": "back",
                    "params": {},
                    "normalized_params": {}
                })
            elif "navigate_home" in extracted_action or "press_home" in extracted_action:
                result.update({
                    "action": "home",
                    "params": {},
                    "normalized_params": {}
                })
            elif "wait" in extracted_action.lower():
                result.update({
                    "action": "wait",
                    "params": {},
                    "normalized_params": {}
                })
            elif "finished" in extracted_action:
                try:
                    text = re.search(r"content='(.*?)'", extracted_action).group(1)
                except:
                    text = ""
                result.update({
                    "action": "terminate",
                    "params": {"text": text},
                    "normalized_params": {"text": text}
                })
            elif "open" in extracted_action.lower():
                try:
                    text = re.search(r"content='(.*?)'", extracted_action).group(1)
                except:
                    text = ""
                result.update({
                    "action": "open",
                    "params": {"app_name": text},
                    "normalized_params": {"app_name": text},
                })
            else:
                raise ValueError(f"Invalid action: {extracted_action}")
        except:
            result.update({
                "action": "invalid",
                "params": {},
                "normalized_params": {}               
            })
        return result


class qwen2vl_Wrapper():


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
    self.client=OpenAI_Client("10.221.105.108", port=42304)
    self.message_handler = qwen2vl_message_handler()
    self.message = []




  def predict_mm(self, goal, current_image_path, history):

    req_messages = self.message_handler.process_message(goal,current_image_path,history)
    response = self.client.call(req_messages, temparature=0, max_tokens=512, top_p=0.9)
    output = self.message_handler.process_response(response,1080,2400)
    print("######resp#############")
    print(response)
    print("######output#############")
    print(output)


    return response, output
