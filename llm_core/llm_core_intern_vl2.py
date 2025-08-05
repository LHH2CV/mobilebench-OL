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
PROMPT_PREFIX = (
    'You are an agent who can operate an Android phone on behalf of a user.'
    " Based on user's goal/request, you may\n"
    '- Complete some tasks described in the request by performing actions on the phone using visual understanding.\n\n'
    'At each step, you will be given the history path, current screenshot (before action) and the task goal.\n'
    'You must analyze the screen and output your action decision in two parts:\n'
    '1. A brief reasoning in Chinese: Why and where to take the next action.\n'
    '2. A structured action command in format below.\n\n'
    'Supported Actions:\n'
    '- Click/tap a position on screen: `{{"click(start_point=<x,y>)"}}`\n'
    '- Scroll the screen: `scroll(start_point="<x1,y1>", direction="up/down/left/right")`}}`\n'
    '- Type text into an input field when searching: `{{"type(content=...)''}}`\n'
   # '- Open an app: `{{"action_type": "open_app", "app_name": "<name>"}}`\n'
    '- Press home button: `{{press_home()}}`\n'
    '- Press back button: `{{press_back()}}`\n'
    '- Wait for UI update: `{{wait()}}`\n'
    '- The task is finished: `{{finished(content='')}}`\n'
    'You must only use the above 7 actions. \n'
    'Use coordinates based on your visual understanding of the screenshot.\n'
)



GUIDANCE = (
    'Here are some useful guidelines you need to follow:\n'
    'General:\n'
    '- Usually there will be multiple ways to complete a task, pick the'
    ' easiest one. Also when something does not work as expected (due'
    ' to various reasons), sometimes a simple retry can solve the problem,'
    " but if it doesn't (you can see that from the history),"
    ' SWITCH to other solutions.\n'
    '- Sometimes you may need to navigate the phone to gather information'
    ' needed to complete the task, for example if user asks'
    ' "what is my schedule tomorrow", then you may want to open the calendar'
    ' app (using the `open_app` action), look up information there, answer'
    " user's question (using the `answer` action) and finish (using"
    ' the `status` action with complete as goal_status).\n'
    '- For requests that are questions (or chat messages), remember to use'
    ' the `answer` action to reply to user explicitly before finish!'
    ' Merely displaying the answer on the screen is NOT sufficient (unless'
    ' the goal is something like "show me ...").\n'
    '- If the desired state is already achieved (e.g., enabling Wi-Fi when'
    " it's already on), you can just complete the task.\n"
    'Action Related:\n'
    '- Use the `open_app` action whenever you want to open an app'
    ' (nothing will happen if the app is not installed), do not use the'
    ' app drawer to open an app unless all other ways have failed.\n'
    '- Use the `input_text` action whenever you want to type'
    ' something (including password) instead of clicking characters on the'
    ' keyboard one by one. Sometimes there is some default text in the text'
    ' field you want to type in, remember to delete them before typing.\n'
    '- For `click`, `long_press` and `input_text`, the index parameter you'
    ' pick must be VISIBLE in the screenshot and also in the UI element'
    ' list given to you (some elements in the list may NOT be visible on'
    ' the screen so you can not interact with them).\n'
    '- Consider exploring the screen by using the `scroll`'
    ' action with different directions to reveal additional content.\n'
    '- The direction parameter for the `scroll` action can be confusing'
    " sometimes as it's opposite to swipe, for example, to view content at the"
    ' bottom, the `scroll` direction should be set to "down". It has been'
    ' observed that you have difficulties in choosing the correct direction, so'
    ' if one does not work, try the opposite as well.\n'
    'Text Related Operations:\n'
    '- Normally to select certain text on the screen: <i> Enter text selection'
    ' mode by long pressing the area where the text is, then some of the words'
    ' near the long press point will be selected (highlighted with two pointers'
    ' indicating the range) and usually a text selection bar will also appear'
    ' with options like `copy`, `paste`, `select all`, etc.'
    ' <ii> Select the exact text you need. Usually the text selected from the'
    ' previous step is NOT the one you want, you need to adjust the'
    ' range by dragging the two pointers. If you want to select all text in'
    ' the text field, simply click the `select all` button in the bar.\n'
    "- At this point, you don't have the ability to drag something around the"
    ' screen, so in general you can not select arbitrary text.\n'
    '- To delete some text: the most traditional way is to place the cursor'
    ' at the right place and use the backspace button in the keyboard to'
    ' delete the characters one by one (can long press the backspace to'
    ' accelerate if there are many to delete). Another approach is to first'
    ' select the text you want to delete, then click the backspace button'
    ' in the keyboard.\n'
    '- To copy some text: first select the exact text you want to copy, which'
    ' usually also brings up the text selection bar, then click the `copy`'
    ' button in bar.\n'
    '- To paste text into a text box, first long press the'
    ' text box, then usually the text selection bar will appear with a'
    ' `paste` button in it.\n'
    '- When typing into a text field, sometimes an auto-complete dropdown'
    ' list will appear. This usually indicating this is a enum field and you'
    ' should try to select the best match by clicking the corresponding one'
    ' in the list.\n'
)


ACTION_SELECTION_PROMPT_TEMPLATE = (
    PROMPT_PREFIX
    + '\nThe current user goal is: {goal}\n\n'
    'Here is a brief summary of what you have done so far:\n{history}\n\n'
    'A screenshot of the current phone UI is provided.\n'
    '{additional_guidelines}'
    + '\nNow output your decision:\n'
    'Thought: （请用中文推理你的意图和位置）\n'
    'Action: ...\n'
)


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

class intern_vl2_message_handler(object):
 
    def process_message(
        self,
        task: str,
        image_path: str,
        history: Optional[Dict[str, List[str]]] = None,
    ) -> List[Dict[str, Any]]:

        history_summaries = ""
        if history:
            response_list = history.get("history_response", [])
            screenshot_list = history.get("history_image_path", [])
            pairs = list(zip(response_list, screenshot_list))[-9:]

            for i, (reply, _) in enumerate(pairs):
                history_summaries += f"[Step {i+1}] {reply.strip()}\n"

        prompt_text = ACTION_SELECTION_PROMPT_TEMPLATE.format(
            goal=task,
            history=history_summaries.strip(),
            additional_guidelines=GUIDANCE
        )
        messages = [
            {
                "role": "system",
                "content": prompt_text
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": action_parser_tool.image_to_uri(image_path)},
                    }
                ],
            }
        ]
        return messages

    def process_response(self, content, width, height):
        result = action_parser_tool.parse_agent_output(content)
        extracted_action = result["action"]
        try:
            if "click" in extracted_action:
                x,y = action_parser_tool.extract_xy_from_point(extracted_action)
                params = {
                    "position":[int(x * width / 1000), int(y * height / 1000)],
                    "click_times":1
                }
                normalized_params = {
                    "position":[round(x/1000,2),round(y/1000,2)],
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
                params = {
                    "position":[int(x/1000 * width), int(y/1000 * height)],
                    "direction":dir
                }
                normalized_params = {
                    "position":[round(x/1000,2),round(y/1000,2)],
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


    


class intern_vl2_Wrapper():


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
    self.client=OpenAI_Client("10.221.105.108", port=42307)
    self.message_handler = intern_vl2_message_handler()




  def predict_mm(self, goal, current_image_path, history):

    req_messages = self.message_handler.process_message(goal,current_image_path,history)
    response = self.client.call(req_messages, temparature=0, max_tokens=512, top_p=0.9)
    output = self.message_handler.process_response(response,1080,2400)
    return response,output
