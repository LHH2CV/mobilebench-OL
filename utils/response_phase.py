from typing import List, Dict, Any, Optional, Tuple
import re

def extract_swipe_points(text: str) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    支持宽松格式：drag(100,200)-(300,400)、swipe from (x1,y1) to (x2,y2)、<point>x y</point> 多种形式。
    """
    text = text.strip().lower()
    nums = list(map(int, re.findall(r"\d{2,4}", text)))  # 找出所有 2~4 位整数

    if len(nums) >= 4:
        x1, y1, x2, y2 = nums[:4]
        return x1, y1, x2, y2
    
    #print(f"[WARN] Unable to extract 2 points from text: '{text}' → Found {len(nums)} number(s)")

    return None 
def extract_swipe_point_direction(text: str) -> set:
    """
    从包含 swipe 操作的文本中抽取 start_point 和 direction。
    支持各种非结构化描述格式，如：
    - swipe(start_point="(750,100)", direction="right")
    - swipe from (100,200) to right
    - direction=down at 300,800
    - swipe towards left
    """
    # 1. 抽取方向（支持各种形式：direction=right、towards right、to right、: right 等）
    direction_pattern = r'(?:direction\s*[:=]\s*|to\s+|towards\s+)?[\'"]?(up|down|left|right)[\'"]?'
    direction_match = re.search(direction_pattern, text, re.IGNORECASE)
    direction = direction_match.group(1).lower() if direction_match else None

    # 2. 抽取坐标（匹配形如 (750, 100)、<750,100>、"750, 100"）
    point_pattern = r'[\(<\s]*\s*(\d{2,4})\s*[,， ]\s*(\d{2,4})\s*[\)>]?'  # 宽松处理中英逗号和空格
    point_match = re.search(point_pattern, text)
    if point_match:
        x, y = int(point_match.group(1)), int(point_match.group(2))
        return x,y,direction
    return None
    
def extract_xy_from_point(text: str) -> Tuple[int, int]:
    """
    从输入字符串中抽取 x, y 坐标。
    支持多种格式：
    - <point>100 120</point>
    - (300,800)
    - point='(100, 200)'
    - click at 200 400
    - x1="300" y1="600"
    - x=300, y=500
    - 连续两个数字（出现在 point 后）
    """
    text = text.strip()

    # 1. <point>100 120</point> 或 <point>100,120</point>
    tag_match = re.search(r'<point>\s*(\d+)[\s,]+(\d+)\s*</point>', text)
    if tag_match:
        return int(tag_match.group(1)), int(tag_match.group(2))

    # 2. x1="123" y1="456" or x1=123 y1=456
    attr_match = re.search(r'x1\s*=\s*["\']?(\d+)', text) and re.search(r'y1\s*=\s*["\']?(\d+)', text)
    if attr_match:
        x1 = int(re.search(r'x1\s*=\s*["\']?(\d+)', text).group(1))
        y1 = int(re.search(r'y1\s*=\s*["\']?(\d+)', text).group(1))
        return x1, y1

    # 3. x=..., y=...（可有可无引号）
    x_match = re.search(r'x\s*=\s*["\']?(\d+)', text)
    y_match = re.search(r'y\s*=\s*["\']?(\d+)', text)
    if x_match and y_match:
        return int(x_match.group(1)), int(y_match.group(1))

    # 4. point=(300, 400) 或 point='(300,400)'
    tuple_match = re.search(r'\(?\s*(\d+)\s*,\s*(\d+)\s*\)?', text)
    if tuple_match:
        return int(tuple_match.group(1)), int(tuple_match.group(2))

    # 5. point 后至少有两个数字：point=..., point 100 200, etc.
    point_idx = text.lower().find("point")
    if point_idx != -1:
        tail = text[point_idx:]
        nums = re.findall(r"\d+", tail)
        if len(nums) >= 2:
            return int(nums[0]), int(nums[1])

    # 6. 整段中寻找连续两个数字（最后手段）
    nums = re.findall(r"\d+", text)
    if len(nums) >= 2:
        return int(nums[0]), int(nums[1])

    raise ValueError("Cannot extract x and y coordinates from input.")
def parse_agent_output(content: str) -> dict:
    try:
        content = content.strip()

        # 提取 Thought 部分
        thought_match = re.search(r"Thought:\s*(.*?)\s*Action:", content, re.DOTALL)
        action_match = re.search(r"Action:\s*(.*)", content, re.DOTALL)

        if not thought_match or not action_match:
            raise ValueError("Missing 'Thought:' or 'Action:' section")

        extracted_thought = thought_match.group(1).strip()
        extracted_action = action_match.group(1).strip()
        cleaned_action = (
            extracted_action
            .replace("'='", "")
            .replace("'\n'", "")
            .replace("'\n", "")
            .replace("((", "(")
            .replace("))", ")")
            .replace("\n", "")
            .strip()
        )

        return {
            "response": content,
            "thought": extracted_thought,
            "action": cleaned_action
        }

    except Exception as e:
        print(f"[ERROR] Failed to parse content:\n{content}\nReason: {e}")
        return {
            "response": content,
            "thought": None,
            "action": None,
            "error": str(e),
            "action_type": "invalid"
        }