# MobileBench-OL

**MobileBench-OL** 是一个面向多模态智能体的 **在线评测 Benchmark**，支持对多模型、多任务进行自动化批量评估，具备可重试机制与状态重置功能。


## 🚀 快速开始

```bash

1.启动主程序：
python maintask.py

2.参数修改（可在 main() 中手动修改）：
RETRY_ROUNDS = 2   # 未成功保存任务的补跑轮次
CONNECT_RETRY = 3  # 手机连接失败重试次数
FAIL_RETRY    = 1  # 单任务失败重试次数
reset        = True   # 是否启用 reset 流程
SERIAL       = "9945aam77ld6y9u4"   # adb 设备序列号 终端输入adb devices获取
MODEL_NAME   = "uitars_1_5_reset_inital_longtail_version0_7_22" # 模型+任务+日期标识
BASE_DIR     = Path("result") / MODEL_NAME   # 轨迹保存路径
task_file    = "top12.csv"  # 任务列表文件

3.模型接入说明
大部分模型通过 OpenAI API 格式（/v1/chat/completions）进行接入，封装在 llm_core_xxx.py 中
若使用 vLLM 启动推理服务，请在 model wrapper 层中自定义修改 IP 与端口。

所有模型的输入由 process_message() 构造，包括：
当前截图 image_path
当前任务目标 task
上下文历史 history（包含 past image + response）

4.apk
稳定版本的apk已上传llmshared/wuqinzhuo/MobileBench-V3/apk（小米融合云https://cloud.mioffice.cn/juicefs/）
可以通过apk_install.py下载到设备

MobileBench-OL/
├── maintask.py                 # 主评测逻辑入口
├── utils/
│   ├── llm_core_xxx.py         # xxx模型 Client 封装
│   ├── agent                   # agent 负责device和model交互
│   ├── adb_excutor.py          # action执行器
│   └── evaluator_xpath.py      # XPath评估工具
├── result/
│   └── uitars_1_5_reset_.../   # 保存评测轨迹
└── top12.csv                   # 示例任务集
