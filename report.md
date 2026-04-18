# MLSYS Course Project Report

## 1. Agent Overview
本项目实现了一个基于 LLM 的 GPU 指标探测 Agent。Agent 的核心目标是：读取评测目标规范，自动生成 CUDA 探测程序，编译运行后输出结构化结果。

核心设计原则：
- 仅通过 LLM 生成 CUDA 测试代码；
- 从 `/target/target_spec.json` 读取评测目标；
- 结果输出为单一 `output.*` 文件；
- 模型接口通过环境变量注入（`API_KEY`、`BASE_MODEL`、`BASE_URL`）。

## 2. System Architecture
项目主要由以下模块组成：
- `agent/agent.py`：主控制流程（读取 target、调用 LLM、生成/编译/执行 CUDA、重试修复、写出结果与诊断信息）；
- `agent/prompts/system_probe.txt`：系统提示词，约束 LLM 代码生成行为与结果自检策略；
- `llm/openai_client.py`：OpenAI 兼容客户端初始化，支持 `.env` 与环境变量；
- `run.sh`：评测入口脚本，执行 `python3 -m agent.agent`。

## 3. Execution Workflow
1. 读取 `/target/target_spec.json`；
2. 基于 target 与系统提示词构建 prompt，请求 LLM 生成 CUDA 代码；
3. 解析 LLM 返回代码块并写入 `.cu` 文件；
4. 使用 `nvcc` 编译并执行二进制；
5. 解析程序输出，抽取目标指标；
6. 若失败（编译/运行/解析异常），将错误上下文反馈给 LLM 并重试；
7. 成功后写入 `/workspace/output.json`，并记录诊断信息（含 LLM trace）。

## 4. Robustness Improvements
为提升稳定性，进行了以下改进：
- 增强代码块解析：支持 `cpp/cuda/c++` fenced block，并清理首行语言标记残留；
- 失败闭环：将编译和运行错误回灌给 LLM，进行迭代修复；
- 可配置重试：支持 `AGENT_MAX_RETRY`，默认 10 轮；
- 诊断可追踪：输出 `.agent_workspace/diagnostics.json` 与 `llm_trace.jsonl` 便于定位失败原因。

## 5. Current Status and Limitations
当前 Agent 可在远端环境中完成“LLM 生成 CUDA → 编译运行 → 产出 output”的主流程。

已观察到的限制：
- 评测端可能出现“任务失败但日志显示单次探测成功”的情况；
- 不同环境下工具链与驱动差异会影响个别指标稳定性；
- 指标定义口径与单位仍需持续对齐评测器预期。

## 6. Conclusion
本 Agent 已实现端到端自动探测能力，并具备错误自修复与可观测性。后续优化方向是继续对齐指标语义与评测器校验规则，提升最终提交成功率与结果一致性。
