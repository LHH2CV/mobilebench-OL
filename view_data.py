      
from datetime import datetime
import time

import streamlit as st
import json
import os
from pathlib import Path
import pandas as pd
from collections import defaultdict
from PIL import Image, ImageDraw
import math


try:
    import fcntl

    HAS_FCNTL = True
except ImportError:
    # Windows doesn't have fcntl
    HAS_FCNTL = False

PAGE_URL = os.getenv("PAGE_URL", "http://10.189.149.105:18000")


def load_manual_check_results(base_dir):
    """加载人工标注结果"""
    check_file = os.path.join(base_dir, "task_info_check.json")
    try:
        if os.path.exists(check_file):
            with open(check_file, 'r', encoding='utf-8') as f:
                # 使用文件锁读取（如果支持）
                if HAS_FCNTL:
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    return json.load(f)
                finally:
                    if HAS_FCNTL:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return {}
    except Exception as e:
        st.error(f"读取标注文件错误: {e}")
        return {}


def generate_success_statistics(data):
    """Generate success statistics by app and task"""

    # Group data by app and task
    app_stats = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'success': 0}))

    for item in data:
        app = item.get('task_app_CHN', 'Unknown')
        task = item.get('task_function', 'Unknown')
        success = item.get('success', False)

        app_stats[app][task]['total'] += 1
        if success:
            app_stats[app][task]['success'] += 1

    return app_stats


def get_episode_unique_id(episode_data):
    """生成episode的唯一标识符"""
    # 使用多个字段组合生成唯一ID
    components = [
        str(episode_data.get('episode_id', '')),
        episode_data.get('app', ''),
        episode_data.get('task', ''),
        episode_data.get('query', ''),
        str(episode_data.get('task_id', ''))
    ]
    return "_".join(filter(None, components))


def save_manual_check_result(base_dir, episode_id, manual_result, auto_result, annotator=""):
    """保存人工标注结果，处理并发操作"""
    check_file = os.path.join(base_dir, "task_info_check.json")

    max_retries = 5
    for attempt in range(max_retries):
        try:
            # 读取现有数据
            existing_data = {}
            if os.path.exists(check_file):
                with open(check_file, 'r', encoding='utf-8') as f:
                    if HAS_FCNTL:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    try:
                        existing_data = json.load(f)
                    except json.JSONDecodeError:
                        existing_data = {}
                    finally:
                        if HAS_FCNTL:
                            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            # 更新数据
            existing_data[episode_id] = {
                "manual_result": manual_result,
                "auto_result": auto_result,
                "consistent": manual_result == auto_result,
                "annotator": annotator,
                "timestamp": datetime.now().isoformat()
            }

            # 写入数据
            with open(check_file, 'w', encoding='utf-8') as f:
                if HAS_FCNTL:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    json.dump(existing_data, f, ensure_ascii=False, indent=2)
                finally:
                    if HAS_FCNTL:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            return True

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.1 * (attempt + 1))  # 递增延时重试
                continue
            else:
                st.error(f"保存标注结果失败: {e}")
                return False

    return False



def display_all_statistics(result_data):
    stats_data = [{
        'Total Tasks': result_data["total_tasks"],
        "Repeat Nums": 10,
        'Pass@1 Rate (%)': result_data["all_pass@1"],
        'Pass@5 Rate (%)': result_data["all_pass@5"],
        'Pass@10 Rate (%)': result_data["all_pass@10"],
    }]

    # Display as DataFrame
    df = pd.DataFrame(stats_data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def display_success_statistics(data):
    # Generate and display statistics
    app_stats = generate_success_statistics(data)

    # Add generate markdown button at the top
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        if st.button("📝 生成Markdown报告", type="primary"):
            st.session_state.show_markdown = True

    # Show markdown report if requested
    if getattr(st.session_state, 'show_markdown', False):
        markdown_report = generate_markdown_report(data)

        st.subheader("📄 Markdown报告")

        # Show markdown content in expandable text area
        with st.expander("📋 复制Markdown内容", expanded=True):
            st.text_area(
                "Markdown内容 (复制此内容):",
                value=markdown_report,
                height=300,
                help="复制此Markdown内容用于文档"
            )

        # Show rendered markdown
        with st.expander("📖 渲染预览", expanded=True):
            st.markdown(markdown_report)

        # Reset button
        if st.button("🔄 隐藏Markdown报告"):
            st.session_state.show_markdown = False
            st.rerun()

        st.divider()

    for app_name, tasks in app_stats.items():
        # with st.expander(f"📱 {app_name} - Task Success Statistics", expanded=False):
        st.subheader(f"📱 {app_name} - Task Success Statistics")
        # Prepare data for DataFrame
        stats_data = []
        total_tests = 0
        total_success = 0

        for task_name, stats in tasks.items():
            success_rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
            stats_data.append({
                'Task': task_name,
                'Total Tests': stats['total'],
                'Successful': stats['success'],
                'Success Rate (%)': f"{success_rate:.1f}%"
            })
            total_tests += stats['total']
            total_success += stats['success']

        # Add overall statistics
        overall_success_rate = (total_success / total_tests * 100) if total_tests > 0 else 0
        stats_data.append({
            'Task': 'OVERALL',
            'Total Tests': total_tests,
            'Successful': total_success,
            'Success Rate (%)': f"{overall_success_rate:.1f}%"
        })

        # Display as DataFrame
        df = pd.DataFrame(stats_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Display summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总任务数", len(tasks))
        with col2:
            st.metric("总测试数", total_tests)
        with col3:
            st.metric("总体成功率", f"{overall_success_rate:.1f}%")


def generate_manual_annotation_statistics(data, manual_check_results):
    """Generate manual annotation statistics by app and task"""

    # Group data by app and task
    app_stats = defaultdict(lambda: defaultdict(lambda: {
        'total': 0,
        'auto_success': 0,
        'manual_success': 0,
        'annotated': 0,
        'consistent': 0,
        'inconsistent': 0
    }))

    for item in data:
        app = item.get('app', 'Unknown')
        task = item.get('task', 'Unknown')
        auto_success = item.get('success', False)
        episode_id = get_episode_unique_id(item)

        stats = app_stats[app][task]
        stats['total'] += 1

        if auto_success:
            stats['auto_success'] += 1

        # Check manual annotation
        manual_check = manual_check_results.get(episode_id, {})
        if manual_check:
            stats['annotated'] += 1
            manual_result = manual_check.get('manual_result', False)

            if manual_result:
                stats['manual_success'] += 1

            # Check consistency
            if manual_check.get('consistent', True):
                stats['consistent'] += 1
            else:
                stats['inconsistent'] += 1

    return app_stats

def generate_markdown_report(data):
    """Generate markdown format report for all apps"""
    app_stats = generate_success_statistics(data)

    markdown_content = "# 📊 Test Success Statistics Report\n\n"

    # Overall summary
    total_episodes = len(data)
    total_success = sum(1 for item in data if item.get('success', False))
    overall_rate = (total_success / total_episodes * 100) if total_episodes > 0 else 0

    markdown_content += f"## 📈 Overall Summary\n\n"
    markdown_content += f"- **Total Episodes**: {total_episodes}\n"
    markdown_content += f"- **Successful Episodes**: {total_success}\n"
    markdown_content += f"- **Overall Success Rate**: {overall_rate:.1f}%\n\n"

    # Per app statistics
    for app_name, tasks in app_stats.items():
        markdown_content += f"## 📱 {app_name}\n\n"

        # Create markdown table
        markdown_content += "| Task | Total Tests | Successful | Success Rate (%) |\n"
        markdown_content += "|------|-------------|------------|-----------------|\n"

        total_tests = 0
        total_success = 0

        for task_name, stats in tasks.items():
            success_rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
            markdown_content += f"| {task_name} | {stats['total']} | {stats['success']} | {success_rate:.1f}% |\n"
            total_tests += stats['total']
            total_success += stats['success']

        print("total_success", total_success)
        # Add overall row for this app
        overall_success_rate = (total_success / total_tests * 100) if total_tests > 0 else 0
        markdown_content += f"| **OVERALL** | **{total_tests}** | **{total_success}** | **{overall_success_rate:.1f}%** |\n\n"

        # Add summary metrics
        markdown_content += f"**Summary for {app_name}:**\n"
        markdown_content += f"- Total Tasks: {len(tasks)}\n"
        markdown_content += f"- Total Tests: {total_tests}\n"
        markdown_content += f"- Overall Success Rate: {overall_success_rate:.1f}%\n\n"
        markdown_content += "---\n\n"

    return markdown_content

def display_manual_annotation_statistics(data, manual_check_results):
    """Display manual annotation statistics"""

    # Generate and display statistics
    app_stats = generate_manual_annotation_statistics(data, manual_check_results)

    if not any(stats for app_tasks in app_stats.values() for stats in app_tasks.values() if stats['annotated'] > 0):
        st.info("暂无人工标注数据")
        return

    for app_name, tasks in app_stats.items():
        # 过滤出有标注数据的任务
        annotated_tasks = {task: stats for task, stats in tasks.items() if stats['annotated'] > 0}

        if not annotated_tasks:
            continue

        st.subheader(f"📱 {app_name} - 人工标注统计")

        # Prepare data for DataFrame
        stats_data = []
        total_tests = 0
        total_annotated = 0
        total_auto_success = 0
        total_manual_success = 0
        total_consistent = 0
        total_inconsistent = 0

        for task_name, stats in annotated_tasks.items():
            if stats['annotated'] > 0:  # 只显示有标注数据的任务
                auto_success_rate = (stats['auto_success'] / stats['total'] * 100) if stats['total'] > 0 else 0
                manual_success_rate = (stats['manual_success'] / stats['annotated'] * 100) if stats[
                                                                                                  'annotated'] > 0 else 0
                annotation_coverage = (stats['annotated'] / stats['total'] * 100) if stats['total'] > 0 else 0
                consistency_rate = (stats['consistent'] / stats['annotated'] * 100) if stats['annotated'] > 0 else 0

                stats_data.append({
                    '任务': task_name,
                    '总数': stats['total'],
                    '已标注': stats['annotated'],
                    '标注覆盖率': f"{annotation_coverage:.1f}%",
                    '自动成功率': f"{auto_success_rate:.1f}%",
                    '人工成功率': f"{manual_success_rate:.1f}%",
                    '一致性': f"{consistency_rate:.1f}%",
                    '一致数': stats['consistent'],
                    '不一致数': stats['inconsistent']
                })

                total_tests += stats['total']
                total_annotated += stats['annotated']
                total_auto_success += stats['auto_success']
                total_manual_success += stats['manual_success']
                total_consistent += stats['consistent']
                total_inconsistent += stats['inconsistent']

        if stats_data:  # 只有在有数据时才显示
            # Add overall statistics
            overall_auto_success_rate = (total_auto_success / total_tests * 100) if total_tests > 0 else 0
            overall_manual_success_rate = (total_manual_success / total_annotated * 100) if total_annotated > 0 else 0
            overall_annotation_coverage = (total_annotated / total_tests * 100) if total_tests > 0 else 0
            overall_consistency_rate = (total_consistent / total_annotated * 100) if total_annotated > 0 else 0

            stats_data.append({
                '任务': 'OVERALL',
                '总数': total_tests,
                '已标注': total_annotated,
                '标注覆盖率': f"{overall_annotation_coverage:.1f}%",
                '自动成功率': f"{overall_auto_success_rate:.1f}%",
                '人工成功率': f"{overall_manual_success_rate:.1f}%",
                '一致性': f"{overall_consistency_rate:.1f}%",
                '一致数': total_consistent,
                '不一致数': total_inconsistent
            })

            # Display as DataFrame
            df = pd.DataFrame(stats_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Display summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("标注覆盖率", f"{overall_annotation_coverage:.1f}%")
            with col2:
                st.metric("人工成功率", f"{overall_manual_success_rate:.1f}%")
            with col3:
                st.metric("判断一致性", f"{overall_consistency_rate:.1f}%")
            with col4:
                st.metric("不一致案例", total_inconsistent)


def process_image_with_action(image_path, action, params, normed_pos=True):
    """Process image based on action and params"""

    color = 'red'

    if image_path is None:
        return None

    # check if params is valid
    if (
            "x" in params
            or "direction" in params
            or "start_position" in params
            or "end_position" in params
            or "text" in params
            or "position" in params
    ):
        pass
    else:
        return image_path

    if isinstance(image_path, str):
        # Convert back to PIL for drawing
        img_pil = Image.open(image_path)
    else:
        img_pil = image_path

    width, height = img_pil.size
    draw = ImageDraw.Draw(img_pil)

    # Draw cross cursor if x,y coordinates exist
    if ("x" in params and "y" in params) or ("position" in params):
        if "x" in params and "y" in params:
            x, y = (params["x"]), (params["y"])
        else:
            x, y = (params["position"][0]), (params["position"][1])

        if normed_pos:
            x = int(x * width)
            y = int(y * height)
        else:
            x, y = int(x), int(y)

        if x > width or y > height:
            st.warning(f"x,y坐标超出图片范围: {x}, {y}, {width}, {height}")
        cursor_size = 200
        # Draw red cross
        draw.line(
            [x - cursor_size // 2, y, x + cursor_size // 2, y], fill="red", width=5
        )
        draw.line(
            [x, y - cursor_size // 2, x, y + cursor_size // 2], fill="red", width=5
        )
        # Draw center dot
        dot_radius = 10
        draw.ellipse(
            [x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius], fill="red"
        )

    # Draw bounding box if bbox exists
    if "bbox" in params:
        bbox = params["bbox"]
        if len(bbox) == 2:
            x1, y1 = bbox[0]
            x2, y2 = bbox[1]
            # Draw rectangle with red outline
            draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
        elif len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            # Draw rectangle with red outline
            draw.rectangle([x1, y1, x2, y2], outline="red", width=5)

    # Draw arrow from start_position to end_position if they exist
    if "start_position" in params and "end_position" in params:
        start_x, start_y = int(params["start_position"][0]), int(
            params["start_position"][1]
        )
        end_x, end_y = int(params["end_position"][0]), int(params["end_position"][1])

        # Draw arrow line
        draw.line([start_x, start_y, end_x, end_y], fill="red", width=10)
        # print(f"start_x: {start_x}, start_y: {start_y}, end_x: {end_x}, end_y: {end_y}")

        # Draw arrow head
        arrow_size = 50
        # Calculate angle of line
        angle = math.atan2(end_y - start_y, end_x - start_x)
        # Calculate arrow head points
        x1 = end_x - arrow_size * math.cos(angle + math.pi / 6)
        y1 = end_y - arrow_size * math.sin(angle + math.pi / 6)
        x2 = end_x - arrow_size * math.cos(angle - math.pi / 6)
        y2 = end_y - arrow_size * math.sin(angle - math.pi / 6)

        # Draw arrow head
        draw.polygon([(end_x, end_y), (x1, y1), (x2, y2)], fill="red")

    # Draw direction arrow if direction exists
    elif "direction" in params:
        # Center coordinates
        cx, cy = width // 2, height // 2
        arrow_size = min(width, height) // 8

        direction = params["direction"].lower()
        if direction == "up":
            points = [
                (cx, cy - arrow_size),
                (cx - arrow_size // 2, cy),
                (cx + arrow_size // 2, cy),
            ]
        elif direction == "down":
            points = [
                (cx, cy + arrow_size),
                (cx - arrow_size // 2, cy),
                (cx + arrow_size // 2, cy),
            ]
        elif direction == "left":
            points = [
                (cx - arrow_size, cy),
                (cx, cy - arrow_size // 2),
                (cx, cy + arrow_size // 2),
            ]
        elif direction == "right":
            points = [
                (cx + arrow_size, cy),
                (cx, cy - arrow_size // 2),
                (cx, cy + arrow_size // 2),
            ]

        draw.polygon(points, fill="red")

    return img_pil


def load_jsonl_data(file_path):
    """Load data from jsonl file"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return []
    return data


import json

def load_json_file(file_path):
    """读取JSON文件并返回内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return None


def normalize_path(path_str, base_dir):
    """Normalize path to handle Windows/Linux path formats"""
    if not path_str:
        return None

    # Remove leading ./ or .\
    if path_str.startswith('./') or path_str.startswith('.\\'):
        path_str = path_str[2:]

    # Replace backslashes with forward slashes for consistency
    path_str = path_str.replace('\\', '/')

    path_str = '/'.join(path_str.split('/')[-2:])

    # Join with base directory
    full_path = os.path.join(base_dir, path_str)
    png_path = full_path.replace('.jpg', '.png')
    if not os.path.exists(png_path):
        return full_path
    return png_path


# def filter_data(data, app_filter, task_filter, task_id_filter, query_filter, episode_id_filter):
#     """Filter data based on various criteria"""
#     filtered_data = []
#
#     for item in data:
#         # Apply filters
#         if app_filter and app_filter.lower() not in item.get('app', '').lower():
#             continue
#         if task_filter and task_filter.lower() not in item.get('task', '').lower():
#             continue
#         if task_id_filter is not None and item.get('task_id') != task_id_filter:
#             continue
#         if query_filter and query_filter.lower() not in item.get('query', '').lower():
#             continue
#         if episode_id_filter and episode_id_filter.lower() not in item.get('episode_id', '').lower():
#             continue
#
#         filtered_data.append(item)
#
#     return filtered_data
def filter_data(data, app_filter, task_filter, success_filter, task_id_filter, repeat_nums):
    """Filter data based on various criteria"""
    filtered_data = []
    for task in data:
        tmp = []
        for i in range(repeat_nums):
            item = task[i]
            if app_filter and app_filter.lower() not in item.get('task_app_CHN', '').lower():
                continue
            if task_filter and task_filter.lower() not in item.get('task_function', '').lower():
                continue
            if success_filter is not None and bool(item.get('success')) != success_filter:
                continue
            if task_id_filter is not None and item.get('task_id') != task_id_filter:
                continue
            tmp.append(item)
        if len(tmp) > 0:
            filtered_data.append(tmp)

    return filtered_data


def display_step_data(step_data, base_dir):
    """Display a single step's data with image and annotations"""
    col1, col2 = st.columns([1, 2])

    with col1:
        # Display screenshot
        screenshot_path = step_data.get('screenshot')
        if screenshot_path:
            normalized_path = screenshot_path
            if normalized_path and os.path.exists(normalized_path):
                try:
                    func = step_data["action"]["action"]
                    params = {}
                    if func.lower() in ['click', 'tap']:
                        params['x'] = step_data["action"]["params"]["position"][0]
                        params['y'] = step_data["action"]["params"]["position"][1]

                    if params:
                        annotated_image = process_image_with_action(normalized_path, func, params, normed_pos=False)
                        if annotated_image:
                            st.image(annotated_image, caption=normalized_path, use_container_width=True)
                        else:
                            # Fallback to original image
                            st.image(normalized_path, caption=normalized_path, use_container_width=True)
                    else:
                        st.image(normalized_path, caption=normalized_path, use_container_width=True)

                except Exception as e:
                    st.error(f"Error loading image: {e}")
                    st.text(f"Path: {normalized_path}")
            else:
                st.warning(f"Image not found: {normalized_path}")

    with col2:
        # Display step info (excluding screenshot)
        step_info = {k: v for k, v in step_data["action"].items()}
        st.json(step_info)


def main():
    # 初始化管理状态
    if 'selected_repeat_n' not in st.session_state:
        st.session_state.selected_repeat_n = 0

    repeat_nums = 1

    st_query = st.query_params

    st.set_page_config(page_title="高泛化测试集结果可视化", layout="wide")
    st.title("高泛化测试集结果可视化")

    # File input
    st.sidebar.header("Data Input")
    # uploaded_file = st.sidebar.file_uploader("Upload JSONL file", type=['jsonl'])
    uploaded_file = None



    if not uploaded_file:

        def get_subdirectories(folder_path):
            subdirectories = []
            for item in os.listdir(folder_path):
                item_path = os.path.join(folder_path, item)
                if os.path.isdir(item_path):
                    subdirectories.append(item_path)
            return subdirectories
        prefix = r"C:\\Users\\leonic\\Desktop\\GUI-exe\\result"
        models = ["internvl"]
        selected_model = st.sidebar.selectbox("选择模型:", models)
        model_task_folder = {
            selected_model: []
        }
        import re
        def sort_key(dir_name):
            match = re.search(r'_(\d+)$', dir_name)  # 匹配最后的数字
            if match:
                return int(match.group(1))
            else:
                print(f"[WARN] 未找到数字部分: {dir_name}")
                return float('inf')

        # for model in models:
        model_dir = os.path.join(prefix, selected_model)
        all_task_dirs = sorted(os.listdir(model_dir), key=sort_key)
        for sub_folder in all_task_dirs:
            sub_folder_path = os.path.join(model_dir, sub_folder)
            if sub_folder.startswith("task_"):
                model_task_folder[selected_model].append(sub_folder_path)


        data = []
        target_model_dir = model_task_folder[selected_model]
        for task in target_model_dir:
            repeat_data = []
            for repeat_n in range(repeat_nums):
               # target_model_repeat_dir = os.path.join(task, f"repeat_{repeat_n + 1}")
                target_model_repeat_task_path = os.path.join(target_model_dir, "trajectory.json")
                with open(target_model_repeat_task_path, 'r', encoding='utf-8') as file:
                    chain = json.load(file)
                repeat_data.append(chain)
            data.extend(repeat_data)


    else:
        # Handle uploaded file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(uploaded_file.getvalue().decode('utf-8'))
            temp_path = f.name

        data = load_jsonl_data(temp_path)
        base_dir = os.path.dirname(temp_path)
        os.unlink(temp_path)


    if not data:
        st.warning("No data loaded. Please upload a JSONL file or check the file path.")
        return

    print(len(data))
    st.success(f"Loaded {len(data) // repeat_nums} tasks (repeat {repeat_nums} times)")


    # Success Statistics
    st.header("📊 APP数据统计")
    with st.expander("APP数据统计", expanded=False):
        display_success_statistics(data)


    # # Filters
    # st.sidebar.header("Filters")
    def_app_filter = st_query.get("task_app_CHN", None)
    def_app_filter = None if def_app_filter == "None" else def_app_filter
    def_task_filter = st_query.get("task_function", None)
    def_task_filter = None if def_task_filter == "None" else def_task_filter
    def_success_filter = int(st_query.get("success", -1))
    def_task_id_filter = st_query.get("task_id", None)
    def_task_id_filter = None if def_task_id_filter == "None" else def_task_id_filter

    # # Get unique values for dropdowns
    apps = list(set([item.get('task_app_CHN', '') for item in data if item.get('task_app_CHN')]))
    tasks = list(set([item.get('task_function', '') for item in data if item.get('task_function')]))
    task_ids = list(set([item.get('task_id') for item in data if item.get('task_id') is not None]))


    app_filter = st.sidebar.selectbox("按应用筛选:", [""] + apps,
                                      index=apps.index(def_app_filter) + 1 if def_app_filter else 0)
    task_filter = st.sidebar.selectbox("按任务筛选:", [""] + tasks,
                                       index=tasks.index(def_task_filter) + 1 if def_task_filter else 0)
    success_filter = st.sidebar.selectbox("按成功状态筛选:", [None, True, False],
                                          index=0 if def_success_filter == -1 else (
                                              1 if def_success_filter == 1 else 0))
    task_id_filter = st.sidebar.selectbox("按任务ID筛选:", [None] + task_ids,
                                          index=task_ids.index(def_task_id_filter) + 1 if def_task_id_filter else 0)
    # query_filter = st.sidebar.text_input("按查询关键词筛选:", value=def_query_filter)
    # episode_id_filter = st.sidebar.text_input("按Episode ID筛选:", value=def_episode_id_filter)

    # manual_check_options = [None, "仅显示不一致", "仅显示一致", "仅显示已标注", "仅显示未标注"]
    # manual_check_index = 0
    # if def_manual_check_filter and def_manual_check_filter in manual_check_options:
    #     manual_check_index = manual_check_options.index(def_manual_check_filter)
    # manual_check_filter = st.sidebar.selectbox("按人工标注筛选:", manual_check_options, index=manual_check_index)

    # Apply filters
    # filtered_data = filter_data(data, app_filter, task_filter, task_id_filter, query_filter, episode_id_filter)
    # 这里的筛选是针对于任务的, 不针对于重复次数
    new_data = []
    for i in range(len(data) // repeat_nums):
        tmp = []
        for k in range(repeat_nums):
            tmp.append(data[i * repeat_nums + k])
        new_data.append(tmp)


    filtered_data = filter_data(new_data, app_filter, task_filter, success_filter, task_id_filter, repeat_nums)

    print("筛选后的数据:", len(filtered_data))
    if not filtered_data:
        st.warning("No data matches the current filters.")
        return

    st.info(f"Showing {len(filtered_data)} episodes after filtering")


    # Data selection
    st.sidebar.header("选择要查看的具体轨迹")
    selected_repeat_n = st.sidebar.number_input(
        "选择重复次数:",
        min_value=0,
        max_value=repeat_nums-1,
        value=0,
        step=1
    )
    if selected_repeat_n != st.session_state.selected_repeat_n:
        st.session_state.selected_repeat_n = selected_repeat_n
        st.rerun()

    selected_index = st.sidebar.number_input(
        "选择任务ID索引:",
        min_value=0,
        max_value=len(filtered_data)-1,
        value=0,
        step=1
    )


    if selected_index < len(filtered_data):
        selected_item = filtered_data[selected_index][selected_repeat_n]
        st.header(f"任务ID {selected_index + 1}: {selected_item.get('query', 'Unknown')}")

        # # 人工标注部分
        # with st.expander("🏷️ 人工标注", expanded=False):
        #
        #     auto_result = selected_item.get('success', False)
        #     existing_check = manual_check_results.get(episode_unique_id, {})
        #     existing_manual_result = existing_check.get('manual_result', None)
        #     existing_annotator = existing_check.get('annotator', default_annotator)
        #     existing_timestamp = existing_check.get('timestamp', '')
        #     is_consistent = existing_check.get('consistent', None)
        #
        #     # 显示当前标注状态
        #     col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
        #
        #     with col1:
        #         st.metric("自动判断结果", "成功" if auto_result else "失败")
        #
        #     with col2:
        #         if existing_manual_result is not None:
        #             st.metric("人工判断结果", "成功" if existing_manual_result else "失败")
        #         else:
        #             st.metric("人工判断结果", "未标注")
        #
        #     with col3:
        #         if is_consistent is not None:
        #             consistency_text = "一致" if is_consistent else "不一致"
        #             if is_consistent:
        #                 st.success(f"结果: {consistency_text}")
        #             else:
        #                 st.error(f"结果: {consistency_text}")
        #         else:
        #             st.info("结果: 未标注")
        #
        #     with col4:
        #         if existing_annotator:
        #             st.info(f"标注人: {existing_annotator}")
        #         else:
        #             st.info("标注人: 无")
        #
        #     # 标注控件
        #     col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
        #
        #     with col1:
        #         annotator_name = st.text_input("标注人姓名:", value=existing_annotator,
        #                                        key=f"annotator_{episode_unique_id}")
        #
        #     with col2:
        #         if st.button("✅ 标注为成功", key=f"success_{episode_unique_id}"):
        #             if annotator_name.strip():
        #                 success = save_manual_check_result(base_dir + "-" + f"case{st.session_state.selected_case+1}",
        #                                                    episode_unique_id, True, auto_result,
        #                                                    annotator_name.strip())
        #                 if success:
        #                     st.success("标注已保存！")
        #                     # 重新加载标注结果
        #                     st.session_state.manual_check_results = load_manual_check_results(base_dir + "-" + f"case{st.session_state.selected_case+1}")
        #                     st.rerun()
        #                 else:
        #                     st.error("保存失败！")
        #             else:
        #                 st.warning("请输入标注人姓名")
        #
        #     with col3:
        #         if st.button("❌ 标注为失败", key=f"fail_{episode_unique_id}"):
        #             if annotator_name.strip():
        #                 success = save_manual_check_result(base_dir + "-" + f"case{st.session_state.selected_case+1}",
        #                                                    episode_unique_id, False, auto_result,
        #                                                    annotator_name.strip())
        #                 if success:
        #                     st.success("标注已保存！")
        #                     # 重新加载标注结果
        #                     st.session_state.manual_check_results = load_manual_check_results(base_dir + "-" + f"case{st.session_state.selected_case+1}")
        #                     st.rerun()
        #                 else:
        #                     st.error("保存失败！")
        #             else:
        #                 st.warning("请输入标注人姓名")
        #
        #     if existing_timestamp:
        #         st.caption(f"最后标注时间: {existing_timestamp}")

        key_info = {
            "query": selected_item.get('task_goal', 'Unknown'),
            "app": selected_item.get('task_app_CHN', 'Unknown'),
            "task": selected_item.get('task', 'Unknown'),
            "success": selected_item.get('success', False),
        }
        if key_info['success']:
            st.success(json.dumps(key_info, indent=2, ensure_ascii=False))
        else:
            st.error(json.dumps(key_info, indent=2, ensure_ascii=False))

        with st.expander("Episode Information", expanded=False):
            episode_info = {k: v for k, v in selected_item.items() if k != 'data'}
            st.json(episode_info)

        ### 这里把每个展示的selected_item数据重新组织下
        selected_item['data'] = []
        history_action, history_image_path, history_response = selected_item["history_action"], selected_item["history_image_path"], selected_item["history_response"]
        for step, (action, image_path, response) in enumerate(zip(history_action, history_image_path, history_response)):
            selected_item['data'].append({
                "step": step + 1,
                "action": action,
                "screenshot": image_path,
                "response": response,
            })

            # Display data steps
        if 'data' in selected_item and selected_item['data']:
            st.header("Steps")

            for i, step_data in enumerate(selected_item['data']):
                display_step_data(step_data, os.path.join(prefix, selected_model))
                st.divider()
        else:
            st.warning("No step data found for this episode.")


if __name__ == "__main__":
    main()


    