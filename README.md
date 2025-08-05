# MobileBench-OL

**MobileBench-OL** is an online evaluation benchmark designed for multimodal agents. It supports automated batch evaluation across multiple models and tasks, with built-in retry mechanisms and environment reset capabilities.


## Requirements and Installation

This work has been tested in the following environment:
* `python == 3.9.12`
* `lxml==5.3.1`
* `uiautomator2==3.2.8`
* `requests==2.25.0`



## Supported Models

| Model             | Model Name                                 | Organization |Address |
|-------------------|--------------------------------------------|--------------|--------------|
| UI-TARS-1.5       | `ui-tars-1.5-7b`                           | Bytedance    |https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B |
| UI-TARS           | `ui-tars-7b` | Bytedance    |https://huggingface.co/ByteDance-Seed/UI-TARS-7B-SFT |
| Qwen2.5-VL        | `qwen2.5-vl-3/7b-instruct`                 | Alibaba      | https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct 
| Qwen2‑VL     | `qwen2‑vl‑7b‑instruct`  | Alibaba                 | https://huggingface.co/Qwen/Qwen2‑VL‑7B‑Instruct
| CogAgent     | `cogagent-9b-20241220`   | THUDM / Z.ai         | https://huggingface.co/zai-org/cogagent-9b-20241220  |       
| UGround‑V1   | `UGround-V1-7B`  | OSU NLP Group | https://huggingface.co/osunlp/UGround-V1-7B         |
| DeepSeek‑VL2 | `deepseek-vl2`   | DeepSeek‑AI   | https://huggingface.co/deepseek-ai/deepseek-vl2-small |
| InternVL2 | `InternVL2-8B`               | OpenGVLab          |      https://huggingface.co/OpenGVLab/InternVL2-8B       |
| OS‑Atlas  | `OS-Atlas-Pro-7B`           | OS‑Copilot       | https://huggingface.co/OS-Copilot/OS-Atlas-Pro-7B

## Command to Start!

```bash

python main_task.py --model_name uitars_1_5 --serial 12345678 --task_file top12.csv --trajectory_file round1
python main_task.py --model_name uitars_1_5 --serial 12345678 --task_file longtail-reset.csv --trajectory_file round1
```

**Arguments**
- `--retry_rounds (int, default to 2)`: Number of retry rounds for tasks that failed to save successfully.
- `--connect_retry (int, default to 3)`: Number of retries when failing to connect to the device or environment.
- `--fail_retry (int, default to 1)`: Number of retries for individual task failures.
- `--reset(bool)`: Use this flag to evaluate the reset task set. If not set, only regular tasks will be evaluated.
- `--serial (str, default to '12345678')` : Android device serial number (can be found using adb devices).
- `--model_name (str, default to 'test')`: Model name(selected in `uitars_1_5,uitars,gpt4o,cogagent,os_altas,qwen2.5vl,Qwen 2.5-VL,qwen2vl,uground,deepseek,intern,React_gpt4o,React_deepseek,React_uitars_1_5`).
- `--task_file (str, default to 'top12.csv')`: CSV file specifying the list of tasks to run(top12.csv,top12-reset.csv,lontail.csv,longtial-reset.csv).
- `--trajectory_file (str, default to 'test')`: Sub-directory or file name to store trajectory results under the result folder.

3.模型接入说明
大部分模型通过 OpenAI API 格式（/v1/chat/completions）进行接入，封装在 llm_core_xxx.py 中
若使用 vLLM 启动推理服务，请在 model wrapper 层中自定义修改 IP 与端口。

所有模型的输入由 process_message() 构造，包括：
当前截图 image_path
当前任务目标 task
上下文历史 history（包含 past image + response）


## APKs
The stable version of the APK has been uploaded to:
llmshared/wuqinzhuo/MobileBench-V3/apk (Xiaomi Cloud Fusion: https://cloud.mioffice.cn/juicefs/)
You can use apk_install.py to download and install it on your device.

```
MobileBench-OL/
├── maintask.py                 # Entry point for the main evaluation logic
├── llm_core/
│   ├── llm_core_xxx.py        
│   └── llm_core_xxx.py         # Model-specific LLM client wrappers
├── utils/
│   ├── agent                   # Agent handling interaction between device and model
│   ├── adb_excutor.py          # Action executor for GUI operations
│   ├── tool_xxx.py             # Utility modules
│   └── evaluator_xpath.py      # XPath-based evaluation tools
├── result/
│   └── uitars_1_5_reset_.../   # Stores evaluation trajectories
└── top12.csv                   # Example task set
```



## Devices ID

Please use adb connect to connect a online mobile phone to your computer. Or use USB to connect a physic mobile phone to your computer.

Use adb devices to check the connected Devices ID.

![image](picture/device_id.png)

Replace the Devices ID in the Arguments.
