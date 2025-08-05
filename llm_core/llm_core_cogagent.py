import openai
from openai import OpenAI
import base64
from typing import List, Dict, Any, Optional, Tuple
import re
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

class cogagent_message_handler(object):
    
    def process_message(self,task: str,image_path: str,history: dict
    ) -> List[Dict[str, Any]]:
        current_platform = "Mobile"
        platform_str = f"(Platform: {current_platform})\n"
        format_str = "(Answer in Status-Plan-Action-Operation-Sensitive format.)\n"
        history_str = "\nHistory steps: "
        response_list = history["history_response"]
        if history:
            for i, entry in enumerate(response_list):
                status_action =  re.sub(r'^.*?\bAction:', 'Action:', entry, flags=re.DOTALL)
                history_str += f"\n{i}. {status_action.strip()}"
        query = f"Task: {task}{history_str}\n{platform_str}{format_str}"
        img_url = f"data:image/jpeg;base64,{encode_image(image_path)}"

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": query,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": img_url},
                    },
                ],
            },
        ]
        return messages



    def extract_grounded_operation(self,response: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extracts the grounded operation and action from the response text.

        Parameters:
        - response (str): The model's response text.

        Returns:
        - (step, action) (Tuple[Optional[str], Optional[str]]): Extracted step and action from the response.
        """
        grounded_pattern = r"Grounded Operation:\s*(.*)"
        action_pattern = r"Action:\s*(.*)"

        step = None
        action = None

        matches_history = re.search(grounded_pattern, response)
        matches_actions = re.search(action_pattern, response)
        if matches_history:
            step = matches_history.group(1)
        if matches_actions:
            action = matches_actions.group(1)

        return step, action


    def process_response(self,response: str,
                     width: int,
                     height: int) -> Dict:
        """
        解析 CLICK / TYPE / SCROLL_* / END 指令
        返回 {"action": ..., "params": {...}, "normalized_params": {...}}
        """
        try:
            grounded_pattern = r"Grounded Operation:\s*(.*)"
            extracted_action = None
            matches_history = re.search(grounded_pattern, response)
            if matches_history:
                extracted_action = matches_history.group(1)
            if not extracted_action:
                return {"action": "invalid", "params": {}, "normalized_params": {}}
            s = extracted_action.strip() #AttributeError: 'NoneType' object has no attribute 'strip'

            # 先把注释符号去掉，例如 '#CLICK(' → 'CLICK('
            if s.startswith("#"):
                s = s.lstrip("#").lstrip()

            op_match = re.match(r"(\w+)", s, flags=re.I)
            if not op_match:
                raise ValueError(f"Invalid action: {extracted_action}")
            op = op_match.group(1).upper()

            # ---------- CLICK ----------
            if op == "CLICK":
                # 1) box=[[x1,y1,x2,y2]]
                m = re.search(r"box=\[\[(\d+),(\d+),(\d+),(\d+)\]\]", s)
                if m:
                    x1, y1, x2, y2 = map(int, m.groups())
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                else:  # 2) start_box=(x,y) 写法
                    m = re.search(r"start_box=[^\d]*(\d+)[^\d]+(\d+)", s)
                    if not m:
                        raise ValueError(f"CLICK without box: {extracted_action}")
                    cx, cy = map(int, m.groups())

                normalized_params = {
                    "position": [cx / 1000, cy / 1000],
                    "click_times": 1,
                }
                params = {
                    "position": [int(cx * width / 1000), int(cy * height / 1000)],
                    "click_times": 1,
                }
                return {
                    "action": "click",
                    "params": params,
                    "normalized_params": normalized_params,
                }

            # ---------- TYPE ----------
            if op == "TYPE":
                text_m = re.search(r"text='(.*?)'", s)
                text = text_m.group(1) if text_m else ""
                return {
                    "action": "type",
                    "params": {"text": text},
                    "normalized_params": {"text": text},
                }

            # ---------- SCROLL_* → swipe ----------
            if op.startswith("SCROLL"):
                # 基于方向定义终点（中心 ± 0.2）
                dir_part = op.split("_")[-1]   # UP / DOWN / LEFT / RIGHT
                cx, cy = 0.5, 0.5              # 起点永远是屏幕中心
                dx = dy = 0.0
                if dir_part == "UP":
                    dy = 0.2
                elif dir_part == "DOWN":
                    dy = -0.2
                elif dir_part == "LEFT":
                    dx = 0.2
                elif dir_part == "RIGHT":
                    dx = -0.2
                else:
                    raise ValueError(f"Unknown SCROLL direction: {extracted_action}")

                sx, sy = cx, cy
                ex, ey = cx + dx, cy + dy

                normalized_params = {
                    "start_position": [sx, sy],
                    "end_position":   [ex, ey],
                    "press_duration": -1,
                }
                params = {
                    "start_position": [int(width * sx),  int(height * sy)],
                    "end_position":   [int(width * ex),  int(height * ey)],
                    "press_duration": -1,
                }
                return {
                    "action": "swipe",
                    "params": params,
                    "normalized_params": normalized_params,
                }

            # ---------- 终止 ----------
            if op.startswith("END") or "finished" in s.lower():
                m = re.search(r"content='(.*?)'", s)
                txt = m.group(1) if m else ""
                return {
                    "action": "terminate",
                    "params": {"text": txt},
                    "normalized_params": {"text": txt},
                }
            else:
                return{
                    "action": "back",
                    "params": {},
                    "normalized_params": {}
                }
        except ValueError as e:
            #except ValueError as e:      # 3⃣ 接住
            print("catch:", e)
        # ---------- 未知 ----------
        #raise ValueError(f"Invalid action: {extracted_action}")

class cogagent_Wrapper():
  """OpenAI GPT4 wrapper.

  Attributes:
    openai_api_key: The class gets the OpenAI api key either explicitly, or
      through env variable in which case just leave this empty.
    max_retry: Max number of retries when some error happens.
    temperature: The temperature parameter in LLM to control result stability.
    model: GPT model to use based on if it is multimodal.
  """

  RETRY_WAITING_SECONDS = 20

  def __init__(
      self,
      max_retry: int = 2,
      temperature: float = 0.0,
      max_length: int = 256,
      url: str = "10.221.105.108",
      port: int = 42307,
  ):

    if max_retry <= 0:
      max_retry = 3
      print('Max_retry must be positive. Reset it to 3')
    self.max_retry = min(max_retry, 5)
    self.temperature = temperature
    self.max_length=max_length
    #self.url=url
    self.client=OpenAI_Client(url, port)
    self.message_handler = cogagent_message_handler()
    self.message = []




  def predict_mm(self, goal, current_image_path, history):

    req_messages = self.message_handler.process_message(goal,current_image_path,history)
    response = self.client.call(req_messages, temparature=0, max_tokens=512, top_p=0.9)
    action_output = self.message_handler.process_response(response=response,width=1080,height=2400)
    return response, action_output
