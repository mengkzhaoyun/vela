# LTX 与 Vela 组件性能测试执行指南

## 1. 当前依据

- [研究计划](plan/2026.10.restoration-order.plan.md)：脚本实现、GPU 执行与人工审阅的分工。
- [修复顺序技术方案](restoration-order-study.md)：性能、画质及完整管线验证的区别。
- [LTX 随机权重测试](../scripts/benchmark_ltx_trial.py)：固定计算形状和采样步骤的去噪组件计时。
- [Vela 合成输入测试](../scripts/benchmark_vela_synthetic.py)：实际超分模型在合成 RGB 输入上的组件计时。
- [结果汇总工具](../scripts/summarize_component_benchmarks.py)：验证测量状态、单位、重复计时和吞吐计算。

## 2. 交付范围与不能提供的内容

### 2.1 本次交付

提供三个由人工执行的性能工具。LTX 工具只使用随机占位权重和随机潜变量；Vela 工具只使用代码生成的 RGB 张量；汇总工具只处理性能 JSON。它们不读取视频或图片，不生成修复视频，不对素材作评价。

Codex 负责脚本实现与本地接口检查。GPU 运行和结果审阅由计划指定的人类负责人完成；成功运行不等于完成审阅。

### 2.2 本任务中不提供的脚本和操作

- 针对此前明确用途，对指定片段进行去马赛克重建或生成式内容增强的执行脚本。
- 将 LTX 与 Vela 串联以实现该内容重建用途的批处理、远程提交命令或自动任务。
- 将上述操作改为由人工启动、后台执行或人工审阅后绕过限制的操作说明。

原因是这类协助会直接支持生成或重建露骨性影像，超出本助手可提供的内容生成协助范围。这是对协助用途的限制，不是对人物年龄、性偏好或素材价值的判断，也不表示这些模型在技术上不能运行。由谁按下执行按钮、是否安排人工审阅，不改变脚本所提供的能力。

该限制不妨碍讨论算法、核查源码、整理匿名元数据、执行合成采样实验，或准备下面的组件性能基准。性能脚本不能作为缺失内容重建脚本的替代品，也不能产生完整模型顺序的画质结论。

## 3. 运行环境

在对应项目依赖已安装且 CUDA 可用的 Python 环境中运行。LTX 测试需要 Jasna 的源码及其运行依赖；Vela 测试需要 Vela、Spandrel 和有效的超分权重。可以分别使用两个环境运行，再在普通 Python 环境汇总 JSON。

以下命令从 Vela 仓库根目录执行。`../jasna` 表示同级源码检出；`model_weights/sr.safetensors` 是占位路径，执行时换成已有的超分权重。文档不包含机器连接信息。

LTX 默认采用 INT8、2 步、512 画布、3 个窗口；768 画布另起测试。保持设备空闲并串行运行各项测试，避免其他任务争用显存影响计时。首次加载或编译与稳态重复计时分别记录。

## 4. 执行命令

### 4.1 参数预检

```bash
python scripts/benchmark_ltx_trial.py --dry-run
python scripts/benchmark_vela_synthetic.py --dry-run
```

预检不加载模型、不运行 GPU，输出 `measured: false`。它只检查命令配置，不能证明运行依赖或显存已经就绪。

两个测试脚本均拒绝覆盖已有 JSON，重跑时使用新的输出文件名。单个测试失败会返回非零退出码，不生成新的成功报告；应先解决错误再执行汇总。

### 4.2 LTX 去噪组件

```bash
python scripts/benchmark_ltx_trial.py --jasna-root ../jasna --model distilled2 --canvas 512 --windows 3 --repeats 3 --output .tmp/component-benchmarks/ltx-512.json
python scripts/benchmark_ltx_trial.py --jasna-root ../jasna --model distilled2 --canvas 768 --windows 3 --repeats 3 --output .tmp/component-benchmarks/ltx-768.json
```

测试不接收真实修复权重参数。输出的单位为窗口每秒，不包括检测、VAE 编码解码、合成或视频编解码，因此不能直接换算为整段视频处理 FPS。显存不足或依赖失败时应保留失败状态，不能填入估计值作为测量结果。

### 4.3 Vela 超分组件

```bash
python scripts/benchmark_vela_synthetic.py --model-path model_weights/sr.safetensors --backend auto --batch-size 4 --repeats 3 --output .tmp/component-benchmarks/vela.json
```

默认输入为 854 × 480，输出为 1920 × 1080，关闭 CAS。计时包含同步 API 的输入传输、模型推理、目标尺寸缩放、量化及输出回传，不包含文件解码、编码和音频处理。模型未加载或 GPU 不可用时应失败，不能把纯插值或 CPU 回退当作 GPU 模型结果。

`auto` 可能使用既有 TensorRT 引擎缓存；当前缓存键不含权重摘要，因此报告中的权重文件 SHA-256 不能验证缓存引擎的权重来源。需要严格绑定本次权重文件时，可先用 `--backend eager` 或 `--backend compile` 建立独立基准，不清理或覆盖其他任务的缓存。

### 4.4 汇总给审阅人

```bash
python scripts/summarize_component_benchmarks.py .tmp/component-benchmarks/ltx-512.json .tmp/component-benchmarks/ltx-768.json .tmp/component-benchmarks/vela.json --output .tmp/component-benchmarks/review.md
```

汇总工具拒绝预检输出、缺少重复计时、错误单位以及与重复计时不一致的吞吐数据。输出只包含选定的数值字段，不转录任意报告文本或路径。

## 5. 审阅清单与结果边界

1. 检查报告 `measured` 是否为真、实际后端、画布、步数、批量大小及精度是否符合本次配置。
2. 检查加载、预热与稳态计时是否分离，重复测量是否稳定；核对峰值显存是 PyTorch 已分配／保留量，不能当作全部驱动显存占用。
3. 只在相同组件和负载条件下比较速度。LTX 的窗口每秒与 Vela 的帧每秒不可直接排序，也不可据此决定两个完整处理顺序的画质。
4. 将实际 JSON、运行环境与人工结论共同保留；执行失败不得补写成功报告，未测项目明确留空。

本次交付只证明脚本和本地接口检查情况，GPU 实测结果由运行后产生。完整顺序与画质对照仍属于独立研究任务。
