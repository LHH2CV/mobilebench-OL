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
## Output Format
```
Thought: ...
Action: ...
```
## Action Space
click(point='<point>x1 y1</point>')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(point='<point>x1 y1</point>', direction='down or up or right or left')
press_home()
press_back()
wait()
finished(content='xxx') # # Submit the task regardless of whether it succeeds or fails.

## Note
- Use Chinese in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
"""
PROMPT_PREFIX = (
    'You are an agent who can operate an Android phone on behalf of a user.'
    " Based on user's goal/request, you may\n"
    '- Complete some tasks described in the request by performing actions on the phone using visual understanding.\n\n'
    'At each step, you will be given the history path, current screenshot (before action) and the task goal.\n'
    'You must analyze the screen and output your action decision:\n'
    '1. A brief reasoning in Chinese: Why and where to take the next action.\n'
    '2. A structured action command in format below.\n\n'
    'Supported Actions:\n'
    '- Click/tap a position on screen: `{{"click(start_point=(x1,y1))"}}`\n'
    '- Scroll the screen: `{{"scroll(start_box=(x1,y1), end_box=(x2,y2))"}}`\n'
    '- Type text into an input field when searching: `{{"type(content=...)''}}`\n'
   # '- Open an app: `{{"action_type": "open_app", "app_name": "<name>"}}`\n'
    '- Press home button: `{{press_home()}}`\n'
    '- Press back button: `{{press_back()}}`\n'
    '- Wait for UI update: `{{wait()}}`\n'
    '- The task is finished: `{{finished(content='')}}`\n'
    'You must only use the above 7 actions. \n'
    'Use coordinates based on your visual understanding of the screenshot.\n'
)

SUMMARY_PROMPT_TEMPLATE = (
    PROMPT_PREFIX
    + '\nThe (overall) user goal/request is: {goal}\n'
    'Now I want you to summerize the latest step.\n'
    'You will be given the screenshot before you performed the action (which'
    ' has a text label "before" on the bottom right), the action you chose'
    ' (together with the reason) and the screenshot after the action was'
    ' performed (which has a text label "after" on the bottom right).\n'
    'Also here is the list of detailed information for some UI elements'
    ' in the before screenshot:\n{before_elements}\n'
    'Here is the list for the after screenshot:\n{after_elements}\n'
    'This is the action you picked: {action}\n'
    'Based on the reason: {reason}\n\n'
    'By comparing the two screenshots (plus the UI element lists) and the'
    ' action performed, give a brief summary of this step. This summary'
    ' will be added to action history and used in future action selection,'
    ' so try to include essential information you think that will be most'
    ' useful for future action selections like what you'
    ' intended to do, why, if it worked as expected, if not'
    ' what might be the reason (be critical, the action/reason might be'
    ' wrong), what should/should not be done next and so on. Some more'
    ' rules/tips you should follow:\n'
    '- Keep it short (better less than 50 words) and in a single line\n'
    "- Some actions (like `answer`, `wait`) don't involve screen change,"
    ' you can just assume they work as expected.\n'
    '- Given this summary will be added into action history, it can be used as'
    ' memory to include information that needs to be remembered, or shared'
    ' between different apps.\n\n'
    'Summary of this step: '
)
MAX_IMAGE_COUNT = 10
IMAGE_FACTOR = 28
MIN_PIXELS = 100 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

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

class uitars_1_5_message_handler(object):
    def process_message_som_elements_list(
        self,
        task: str,
        image_path: str,
        xml_string: str,
        history: Optional[Dict[str, List[str]]] = None,
        step_prefix = "" 
    ) -> List[Dict[str, Any]]:
        

        
       # xml_list   = history.get("history_xml_string", [])
        before_ui_elements=representation_utils.xml_dump_to_ui_elements(xml_string)
        before_ui_elements_list = xml_screen_parser_tool._generate_ui_elements_description_list(
            before_ui_elements,
            (1080,2400),
        )
        elements_refer = "the following is the elements relative to the current screen: " + before_ui_elements_list
        sys_prompt_block = {
            "role": "user",
            "content": [
                {"type": "text", "text": sys_prompt + task},
            ],
        }

        before_pixels = np.asarray(Image.open(image_path)).copy()
        for index, ui_element in enumerate(before_ui_elements):
          if m3a_utils.validate_ui_element(ui_element, (1080,2400)):
            m3a_utils.add_ui_element_mark(
                before_pixels,
                ui_element,
                index,
                (1080,2400),
                (0,0,1080,2400),
                0, 
            )
        save_path = f"{step_prefix}_som.png"
        image = Image.fromarray(before_pixels)
        image.save(save_path, format='PNG')

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
            pairs = list(zip(response_list, screenshot_list))[-4:]
        
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
                        "image_url": {"url": action_parser_tool.image_to_uri(before_pixels)},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": action_parser_tool.image_to_uri(image_path)},
                    },
                    {
                        "type": "text",
                        "text": elements_refer
                    }
                ],
            }
        )

        return messages
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
                    },
                    {
                        "type": "text",
                        "text": f"Task: {task}",
                    },
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
                # new_height ,new_width = smart_resize(height,width)
                # normalized_params = {
                #     "position":[x/new_width, y/new_height],
                #     "click_times":1
                # }
                # params = {
                #     "position":[int(x * width / new_width), int(y * height / new_height)],
                #     "click_times":1
                # }
                normalized_params = {
                    "position": [round(x / width, 2), round(y / height, 2)],
                    "click_times":1
                }
                params = {
                    "position":[x, y],
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
                x,y,dir = action_parser_tool.extract_swipe_point_direction(extracted_action)
                #new_height ,new_width = smart_resize(height,width)
                # normalized_params = {
                #     "position":[x/new_width, y/new_height],
                #     "direction":dir
                # }
                # params = {
                #     "position":[int(x * width / new_width), int(y * height / new_height)],
                #     "direction":dir
                # }
                normalized_params = {
                    "position":[round(x / width ,2), round(y / height,2)],
                    "direction":dir
                }
                params = {
                    "position":[x, y],
                    "direction":dir
                }
                result.update({
                    "action": "scroll",
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
    
    def process_message_summary(self,history,after_pixels,after_xml_string,goal):


        before_path = history["history_image_path"][-1]
        before_image = Image.open(before_path)
        before_pixels = np.asarray(before_image).copy()
        before_xml_string = history["history_xml_string"][-1]
        reason = history["history_response"][-1]
        action = history["history_action"][-1]

        before_ui_elements=representation_utils.xml_dump_to_ui_elements(before_xml_string)
        before_ui_elements_list = xml_screen_parser_tool._generate_ui_elements_description_list(
            before_ui_elements,
            (1080,2400),
        )
        after_ui_elements=representation_utils.xml_dump_to_ui_elements(after_xml_string)
        after_ui_elements_list = xml_screen_parser_tool._generate_ui_elements_description_list(
            after_ui_elements, (1080,2400)
        )
        for index, ui_element in enumerate(before_ui_elements):
          if m3a_utils.validate_ui_element(ui_element, (1080,2400)):
            m3a_utils.add_ui_element_mark(
                before_pixels,
                ui_element,
                index,
                (1080,2400),
                (0,0,1080,2400),
                0, 
            )
        for index, ui_element in enumerate(after_ui_elements):
          if m3a_utils.validate_ui_element(ui_element, (1080,2400)):
            m3a_utils.add_ui_element_mark(
                after_pixels,
                ui_element,
                index,
                (1080,2400),
                (0,0,1080,2400),
                0, 
            )
        m3a_utils.add_screenshot_label(before_pixels, 'before')
        m3a_utils.add_screenshot_label(after_pixels, 'after')
        summary_prompt = SUMMARY_PROMPT_TEMPLATE.format(
            goal=goal,
            action=action,
            reason = reason,
            before_elements = before_ui_elements_list,
            after_elements = after_ui_elements_list
        )
        sys_prompt_block = {
            "role": "user",
            "content": [
                {"type": "text", "text": summary_prompt},
            ],
        }
        messages: List[Dict[str, Any]] = [sys_prompt_block]
        messages = [{
                "role": "system",
                "content": "You are a helpful assistant."
        }] + messages
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": action_parser_tool.image_to_uri(before_pixels)},
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": action_parser_tool.image_to_uri(after_pixels)},
                    }
                ],
            }
        )
        return messages

class uitars1_5_Wrapper():


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
    self.client=OpenAI_Client("10.221.105.108", port=42302)
    self.message_handler = uitars_1_5_message_handler()


  def predict_mm_som(self, goal, current_image_path, current_xml_string,history,step_prefix):

    req_messages = self.message_handler.process_message_som_elements_list(goal,current_image_path,current_xml_string,history,step_prefix)
    response = self.client.call(req_messages)
    output = self.message_handler.process_response(response,1080,2400)
    return response, output

  def predict_mm(self, goal, current_image_path,history):

    req_messages = self.message_handler.process_message(goal,current_image_path,history)
    response = self.client.call(req_messages)
    output = self.message_handler.process_response(response,1080,2400)
    return response, output
  
  def summarize(self,history,after_pixels,after_xml_string,goal):
    summary_messages = self.message_handler.process_message_summary(history,after_pixels,after_xml_string,goal)
    response = self.client.call(summary_messages)
    return response