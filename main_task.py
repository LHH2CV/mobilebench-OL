
from __future__ import annotations
import csv, json, os, time, hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

import uiautomator2 as u2
from lxml import etree as ET
from utils import agent
from llm_core import llm_core_cogagent
from llm_core import llm_core_os_altas
from llm_core import llm_core_qwen2_5vl
from llm_core import llm_core_qwen2vl
from llm_core import llm_core_gpt4o
from llm_core import llm_core_uitars_1_5
from llm_core import llm_core_deepseek_vl2
from llm_core import llm_core_intern_vl2
from llm_core import llm_core_uitars
from llm_core import llm_core_uground_vl
from utils import adb_executor
from utils import evaluator_xpath as ev
@dataclass
class Task:
    identifier: str
    goal: str
    home_activity: str
    golden_steps: int
    key_nodes: str
    reset_xpath: str
    reset_query: str
@dataclass
class Trajectory:
    task_id: str
    task_goal: str
    history_action: list        
    history_image_path: list  
    history_response: list   
    summary: str
    success: bool


# ---------- 设备管理 ----------

class DeviceManager:
    def __init__(self, serial: str, max_retry: int = 5):
        self.serial = serial
        self.max_retry = max_retry
        self.d = self._connect()
    def is_uiautomator_alive(self,d) -> bool:
        try:
            d.info  
            return True
        except Exception as e:
            print(f"[DeviceManager] uiautomator2 可能不可用: {e}")
            return False

    def _connect(self):
        for attempt in range(self.max_retry):
            try:
                print(f"[DeviceManager] 尝试连接设备 {self.serial}，第 {attempt+1} 次")
                d = u2.connect(self.serial)
                d.set_input_ime(True)
                if self.is_uiautomator_alive(d): 
                    print("[DeviceManager] 连接成功")
                return d
            except Exception as e:
                print(f"[DeviceManager] 连接失败: {e}")
                time.sleep(2)
        raise RuntimeError(f"无法连接到设备: {self.serial}")

    def reconnect(self):
        print("[DeviceManager] 尝试重新连接设备...")
        self.d = self._connect()

    # ---------- 高层 API ----------
    def reset(self):
        """回桌面 + 清最近任务栏（可选）"""
        self.d.press("home")

    def clear_background(self, excludes: Optional[List[str]] = None):
        """
        彻底杀掉后台应用。
        Args:
            excludes: 不想被杀掉的包名列表，如 ['com.android.systemui']
        """
        self.d.app_stop_all(excludes or [])
        self.d.press("home")  # 杀完再回桌面，保证 UI 稳定

    def launch_app(self, activity: str):
        adb_executor.launch_app(activity, self.d)

    def stop_app(self, package: str):
        self.d.app_stop(package)

# ---------- Agent 工厂 ----------

class AgentFactory:
    @staticmethod
    def create(model_name: str, device):
        if model_name.startswith("uitars_1_5"):
            return agent.base_agent(device,llm_core_uitars_1_5.uitars1_5_Wrapper())  
        elif model_name.startswith("uitars"):
            return agent.base_agent(device,llm_core_uitars.uitars_Wrapper())
        elif model_name.startswith("gpt4o"):
            return agent.base_agent(device, llm_core_gpt4o.GPT4oWrapper())
        elif model_name.startswith("cogagent"):
            return agent.base_agent(device, llm_core_cogagent.cogagent_Wrapper())
        elif model_name.startswith("os_altas"):
            return agent.base_agent(device,llm_core_os_altas.os_altas_Wrapper())
        elif model_name.startswith("qwen2.5vl"):
            return agent.base_agent(device,llm_core_qwen2_5vl.qwen2_5vl_Wrapper())
        elif model_name.startswith("qwen2vl"):
            return agent.base_agent(device,llm_core_qwen2vl.qwen2vl_Wrapper())  
        elif model_name.startswith("uground"):
            return agent.base_agent(device,llm_core_uground_vl.uground_Wrapper())  
        elif model_name.startswith("deepseek"):
            return agent.base_agent(device,llm_core_deepseek_vl2.deepseek_vl2_Wrapper()) 
        elif model_name.startswith("intern"):
            return agent.base_agent(device,llm_core_intern_vl2.intern_vl2_Wrapper())
        raise ValueError(f"Unknown model {model_name}")

# ---------- Task 执行 ----------

class TaskExecutor:
    def __init__(self, device_mgr: DeviceManager, agent):
        self.device_mgr = device_mgr
        self.agent = agent

    def run(self, task: Task, save_dir: Path , reset: bool = False) -> Trajectory:
        self.device_mgr.clear_background()
        time.sleep(5)
        self.device_mgr.launch_app(task.home_activity)
        time.sleep(8)

        self.agent.clear()
        max_steps = min(task.golden_steps * 2, 10)
        os.makedirs(save_dir,exist_ok=True)
        if reset:
            for _ in range(max_steps):
                ok, stepdata = self.agent.step(task.reset_query, path=str(save_dir))
                if ok:
                    break
            success = evaluator_xpath.evaluate(task.reset_xpath, stepdata)
        else:
            for _ in range(max_steps):
                ok, stepdata = self.agent.step(task.goal, path=str(save_dir))
                if ok:
                    break
            time.sleep(3)
            success = evaluator_xpath.evaluate(task.key_nodes, stepdata)

        traj = Trajectory(
            task_id=task.identifier,
            task_goal= task.goal if not reset else task.reset_query,
            history_action=stepdata["history_action"],
            history_image_path=stepdata["history_image_path"],
            history_response=stepdata["history_response"],
            summary=stepdata["summary"],
            success=success,
        )
        return traj


class evaluator_xpath:
    @staticmethod
    def evaluate(task_rule: str,stepdata: dict) -> bool:
        return ev.evaluate(task_rule,stepdata)


class ResultSink:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.cache: Dict[str, bool] = self._load_cache()

    def _load_cache(self):
        fp = self.base_dir / "result_list.txt"
        if not fp.exists():
            return {}
        return {
            line.split(",")[0]: line.split(",")[1].lower() == "true"
            for line in fp.read_text().strip().split(" ")
            if line
        }

    def save(self, traj: Trajectory):
        task_dir = self.base_dir / traj.task_id
        task_dir.mkdir(exist_ok=True)
        with open(task_dir / "trajectory.json", "w", encoding="utf-8") as f:
            json.dump(traj.__dict__, f, ensure_ascii=False, indent=2)
        self.cache[traj.task_id] = traj.success
        self._flush_cache()

    def _flush_cache(self):
        fp = self.base_dir / "result_list.txt"
        fp.write_text(
            " ".join(f"{k},{v}" for k, v in self.cache.items()), encoding="utf-8"
        )

    # 事后评估整轮通过率
    def summary(self):
        if not self.cache:
            return 0.0
        passed = sum(self.cache.values())
        return passed * 100 / len(self.cache)

# ---------- CSV Loader ----------
def load_tasks(csv_path: Path) -> List[Task]:
    tasks: List[Task] = []
    with csv_path.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row["key_nodes"]:
                continue
            tasks.append(
                Task(
                    identifier=row["task_identifier"],
                    goal=row["goal"],
                    home_activity=row["adb_home_page"],
                    golden_steps=int(row["golden_steps"]),
                    key_nodes=row["key_nodes"],
                    reset_xpath = row.get("reset_xpath", "") or "",
                    reset_query = row.get("reset_query", "") or "",
                )
            )
    return tasks



def run_with_reconnect(executor, task, task_dir, reset, dev_mgr, connect_retry=2):
    """
    带连接失败自动重连的任务执行包装器。
    如果抛异常，会尝试最多 connect_retry 次重连后继续执行。
    """
    for attempt in range(connect_retry):
        try:
            return executor.run(task, task_dir, reset)
        except Exception as e:
            print(f"[ERROR] 连接失败（第 {attempt + 1}/{connect_retry} 次）: {e}")
            if attempt < connect_retry - 1:
                print("[INFO] 尝试 reconnect...")
                dev_mgr.reconnect()
                time.sleep(2)
            else:
                raise  # 最后一次仍失败就抛出


def try_execute_task_with_retry(task:Task,BASE_DIR:str,executor:TaskExecutor,dev_mgr:DeviceManager,connent_retry:int,fail_retry:int,reset:bool) -> Optional['Trajectory']:
    """
    包装任务执行逻辑，支持 FAIL_RETRY 次失败重试（非连接错误），
    并保证无论成功失败，最后一次 traj 都能返回。
    """
    task_dir = BASE_DIR / task.identifier

    for fail_attempt in range(fail_retry):
        # 每次重新执行任务前清空旧数据
        if task_dir.exists():
            for f in task_dir.glob("*"):
                try:
                    f.unlink()
                except Exception as e:
                    print(f"[WARN] 删除文件失败: {f}, {e}")

        try:
            traj = run_with_reconnect(executor, task, task_dir, reset, dev_mgr, connect_retry=connent_retry)
        except Exception as e:
            print(f"[FAIL] 连接失败，任务跳过: {e}")
            return None

        if not traj.success:
            print(f"[WARN] 执行失败,检测是否还有重试次数 {fail_attempt + 1}/{fail_retry}")
            if fail_attempt < fail_retry - 1:
                print("[INFO]重试任务...")
                dev_mgr.reconnect()
                time.sleep(2)
                continue
            else:
                print(f"[FAIL] 多次失败仍未成功：{task.identifier}")
                return traj  # 返回失败 traj，供分析或保存
        else:
            return traj

    return None


def main():
    # -------- 模型和任务配置 --------
    RETRY_ROUNDS = 2   # 未成功保存任务的补跑轮次
    CONNECT_RETRY = 3  # 连接失败重试次数
    FAIL_RETRY = 1     # 单任务失败重试次数
    reset = False      # 任务是否为reset集
    SERIAL ="n7emlbbmfyx8eybq" #"9945aam77ld6y9u4"#"orp7u4jrkjnrsw75"
   # SERIAL = "9945aam77ld6y9u4"
    MODEL_NAME = "debug_test" # model + task + date
    BASE_DIR = Path("result") / MODEL_NAME #轨迹存放位置
    task_file = "top12.csv" #任务文件
    
    tasks = load_tasks(Path(task_file))

    # -------- Agent 初始化 --------
    dev_mgr = DeviceManager(SERIAL)
    agent = AgentFactory.create(MODEL_NAME, dev_mgr.d)
    executor = TaskExecutor(dev_mgr, agent)
    sink = ResultSink(BASE_DIR)

    # -------- 多轮补跑逻辑 --------
    for round_id in range(RETRY_ROUNDS):
        print(f"\n[INFO] 第 {round_id + 1} 轮任务执行开始...")
        any_new_success = False

        for task in tasks:
            if task.identifier in sink.cache:
                continue

            traj = try_execute_task_with_retry(task,BASE_DIR,executor,dev_mgr,CONNECT_RETRY,FAIL_RETRY,reset)
            if traj is not None:
                sink.save(traj)
                print(f"[TRAJ SUCCESS] {task.identifier}  ✅")
                any_new_success = True

        if not any_new_success:
            print("[INFO] 本轮没有新任务成功，提前结束补跑")
            break

    # -------- 总结与评估 --------
    print(f"\n✅ Overall pass rate: {sink.summary():.2f}%")
    ev.re_evaluate_all(MODEL_NAME, task_file,reset)


if __name__ == "__main__":
    main()