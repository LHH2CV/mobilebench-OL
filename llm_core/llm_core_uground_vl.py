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
##screensize = (1080,2400)
## Output Format
```
Thought: ...
Action: ...
```
## Action Space

click(point='<point>x1 y1</point>') #example:type(point='<point>542 372</point>')
type(content='') #example:type(content='GUI-Agent')
open_app(app_name=\'\')
swipe(start_point='<point>x1 y1</point>', end_point='<point>x2 y2</point>') 
press_back()
press_enter()
finished(content='xxx') 


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

class uground_message_handler(object):
    
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
                "content": "You are a helpful assistant, with output format:Thought: ...Action: ...."
        }] + messages
        # ------------- 拼接历史 -------------
        if history:
            response_list   = history.get("history_response", [])
            screenshot_list = history.get("history_image_path", [])
        
            # 只保留「回复‑截图」成对数据里的最后 9 条
            pairs = list(zip(response_list, screenshot_list))[-9:]
        
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
        # messages.append(
        #     {
        #         "role": "user",
        #         "content": [
        #             {
        #                 "type": "text",
        #                 "text": task
        #             }
        #         ],
        #     }
        # )
        return messages



    def process_response(self, content, width, height):
        try:
            print("content",content)
            idx = content.index("Action:")
            extracted_thought = content[8:idx].strip()
            excepted_action = content[idx + 7:].strip()
            extracted_action = excepted_action.replace("'='", "").replace("'\n'", "").replace("'\n", "").replace("((","(").replace("))", ")").replace("\n", "")
            result = {
                "response":content,
                "thought": extracted_thought,
            }
        except Exception as e:
            print("Error",e)
            raise ValueError(f"Invalid content: {content}")

        try:
            s = extracted_action.strip().lower()
            POINT_RE = re.compile(
                r"<point>\s*\(?\s*(\d+)\s*[, ]\s*(\d+)\s*\)?\s*</point>",
                flags=re.I,
            )
            # ---------- CLICK ----------
            if s.startswith("click"):
                x,y = action_parser_tool.extract_xy_from_point(extracted_action)
                # m = POINT_RE.search(extracted_action)
                # if not m:
                #     raise ValueError("CLICK without <point>")
                # x, y = map(int, m.groups())
                result.update(
                    action="click",
                    params={"position": [int(x*width/1000), int(y*height/1000)], "click_times": 1},
                    normalized_params={
                        "position": [x / 1000, y / 1000],
                        "click_times": 1,
                    },
                )

            # ---------- SWIPE ----------
            elif s.startswith("swipe"):
                m = re.search(
                    r"start_point='?<point>\s*(\d+)\s+(\d+)\s*</point>'?.*?"
                    r"end_point='?<point>\s*(\d+)\s+(\d+)\s*</point>'?",
                    extracted_action,
                    flags=re.S,
                )
                if not m:
                    raise ValueError("SWIPE without start/end <point>")
                x1, y1, x2, y2 = map(int, m.groups())
                result.update(
                    action="swipe",
                    params={
                        "start_position": [x1*width/ 1000, y1*height/ 1000],
                        "end_position": [x2*width/ 1000, y2*height/ 1000],
                        "press_duration": -1,
                    },
                    normalized_params={
                        "start_position": [x1 / 1000, y1 / 1000,],
                        "end_position": [x2 / 1000, y2 / 1000,],
                        "press_duration": -1,
                    },
                )

            # ---------- BACK ----------
            elif "press_back" in s:
                result.update(action="back", params={}, normalized_params={})

            # ---------- HOME ----------
            elif "press_home" in s or "navigate_home" in s:
                result.update(action="home", params={}, normalized_params={})

            # ---------- TYPE ----------
            elif s.startswith("type"):
                text = re.search(r"content='(.*?)'", extracted_action).group(1)
                result.update(
                    action="type",
                    params={"text": text},
                    normalized_params={"text": text},
                )

            # ---------- OPEN_APP ----------
            elif s.startswith("open_app") or s.startswith("open"):
                # 兼容  open_app(app_name='xxx')  |  OPEN_APP [xxx]
                m = (
                    re.search(r"app_name='(.*?)'", extracted_action)
                    or re.search(r"\[\s*(.*?)\s*\]", extracted_action)
                    or re.search(r"content='(.*?)'", extracted_action)
                )
                app_name = m.group(1).strip() if m else ""
                result.update(
                    action="open",
                    params={"app_name": app_name},
                    normalized_params={"app_name": app_name},
                )

            # ---------- FINISHED / TERMINATE ----------
            elif s.startswith("finished") or "completed" in s:
                txt = re.search(r"content='(.*?)'", extracted_action)
                text = txt.group(1) if txt else ""
                result.update(
                    action="terminate",
                    params={"text": text},
                    normalized_params={"text": text},
                )

            else:
                raise ValueError("unknown action")

        except Exception as e:
            # 解析失败时记录日志并返回空 dict；不中断整体流程
            #logger.warning("parse_action failed: %s  |  raw: %s", e, extracted_action)
            print("parse_action failed")
            result.update(
                action="invalid",
                params={},
                normalized_params={},
            )
            return result

        return result
            #     raise ValueError(f"Invalid action: {extracted_action}")
            # return result

class uground_Wrapper():


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
    self.client=OpenAI_Client("10.221.105.108", port=42301)
    self.message_handler = uground_message_handler()
    self.message = []




  def predict_mm(self, goal, current_image_path, history):

    req_messages = self.message_handler.process_message(goal,current_image_path,history)
    response = self.client.call(req_messages, temparature=0, max_tokens=512, top_p=0.9)
    #response = "'Thought: 未找到搜索结果\nAction:  click\n(500, 71)'"
    output = self.message_handler.process_response(response,1080,2400)
    print("######resp#############")
    print(response)
    print("######output#############")
    print(output)


    return response, output
# llm = uground_Wrapper()
# # # goal = '我想打开应用商店，然后下载微信app'
# goal = '我想进入“我的”'
# # # goal = '当前页面没有我想要的app，我想打开bilibili，我应该需要想右边滑动页面'
# # goal = '我想查看图片中44码的沙色42鞋子的价格，似乎要往右边滑动'
# history = []
# # dict1 = {}
# # # dict1["plan"] = "1. 点击屏幕顶部的搜索框，激活输入状态； 2. 输入搜索关键词“喜羊羊与灰太狼”； 3. 点击搜索按钮，开始搜索； 4. 在搜索结果中找到 并点击“喜羊羊与灰太狼”相关视频。"
# # # dict1["action"] = "在屏幕顶部中央的搜索框中，左键单击以激活输入状态，准备输入搜索关键词。"
# # #("TYPE(box=[[030,036,970,066]], text='喜羊羊与灰太狼', element_type='false', element_info='false')", '在屏幕顶部中央的搜索框中输入“喜羊羊与灰太狼”作为搜索关键词。')
# # #history.append(dict1)
# current_image_path = "scroll.png"
# resp, action_output = llm.predict_mm(
#      goal,current_image_path,history
#  )
