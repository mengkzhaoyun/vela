# Vela 白皮书视觉素材

## 1. 当前依据

- [技术白皮书](../technical-whitepaper.md)：两张架构图与封面候选的嵌入文档。
- [媒体探测与分段](../../vela/pipeline/probe.py)：按时长生成任务。
- [多 GPU 调度](../../vela/pipeline/scheduler.py)：共享队列、独立进程与首音轨合并。
- [流式工作器](../../vela/pipeline/worker.py)：四槽锁页缓冲、处理帧数检查与分段文件。
- [超分器](../../vela/models/upscaler.py)：后端选择、显存内处理及异步 DMA。
- [TensorRT 引擎](../../vela/models/trt_engine.py)：模型导出、流同步、引擎与 Timing Cache。

## 2. 交付素材

日期：2026-10-28。封面 A / B 均为待用户选定的审阅候选。

| 素材 | 用途 |
| --- | --- |
| [vela-overall-architecture.svg](vela-overall-architecture.svg) | 总体架构，1800 × 1180 原生矢量 |
| [vela-technical-architecture.svg](vela-technical-architecture.svg) | 技术架构，1800 × 1500 原生矢量 |
| [vela-whitepaper-cover-vf.png](vela-whitepaper-cover-vf.png) | A 版：VF Studio 底图 + 统一标题排版 |
| [vela-whitepaper-cover-imagegen.png](vela-whitepaper-cover-imagegen.png) | B 版：imagegen API 底图 + 统一标题排版 |
| [vela-whitepaper-cover-vf-art.png](vela-whitepaper-cover-vf-art.png) | A 版未经修改的生成原图 |
| [vela-whitepaper-cover-imagegen-art.png](vela-whitepaper-cover-imagegen-art.png) | B 版未经修改的生成原图 |

两张 SVG 为自包含原生矢量，含 `viewBox`、`title` 和 `desc`，无外部图像、字体下载、脚本或 `foreignObject` 依赖；支持无损缩放。字体依次使用 Microsoft YaHei、Noto Sans CJK SC、sans-serif。

两版封面均由图像生成服务生成概念底图，再以同一版式叠加标题。原图与排版成图均为 **1672 × 941 PNG**。A 请求 4K，B 请求 3840 × 2160，两服务实际返回相同的上述尺寸；未通过插值放大伪装原生 4K。封面中的山水与颗粒变化为概念视觉，不代表实际超分前后对照。

标题为 `VELA`、`技术白皮书`、`面向真实世界视频超分`、`多 GPU 流式引擎架构`；使用 Segoe UI 与微软雅黑，辅以局部暗色遮罩保持对比度。两个服务使用同一创作主题，具体提交提示词分别保留如下。

## 3. 封面生成记录

### 3.1 A 版：VF Studio

- 工作流：用户指定的 `verdantflare-image` 技能，经 Studio `image.create`，单次生成。
- 请求渠道 / 模型：`codex` / `gpt-image-2.5-sunburst`。
- 请求规格：16:9、4K、high；实际规格见第 2 节。
- 服务任务：`imgtask-e02ddc35a0df4b06`。
- 来源项目：`01a18090-88d2-71f8-8435-e19d359c9545`。
- 原图、提示词与审阅记录已保存至来源项目，固定修订 `01a180a2-37b9-71a7-bd1c-8eaea30ec9e4` 已回读核验；候选未选用。
- 下载文件的大小及 SHA-256 与服务端回读一致；图像、正向提示词与标题完成内容评估，`image.rate_content` 返回 `general`。

实际提交提示词：

```text
Create an exquisitely art-directed abstract cover illustration for a serious engineering technical whitepaper about a multi-GPU streaming video super-resolution engine. Wide 16:9 composition, native 4K output if supported. Near-black deep navy background (#10263d), restrained teal optical light (#087f8c) and tiny copper-gold accents. The LEFT 45 percent must be quiet, dark, uncluttered negative space, reserved for typography to be added later. In the RIGHT 55 percent, sculpt a precise cinematic sequence of thin translucent optical frame plates and ordered parallel compute modules: a delicate stream of low-resolution granular film information enters the structure and resolves into crisp crystalline frame edges and fine optical lines. Suggest fidelity-preserving restoration and efficient parallel processing through material and rhythm, not a literal architecture diagram. Sophisticated editorial design, museum-quality studio lighting, physically plausible glass refractions, brushed dark metal, highly controlled highlights, subtle dimensional depth, impeccable edges, balanced negative space, premium print publication finish. Avoid cyberpunk clutter, generic circuit-board patterns, broad neon haze, floating UI, graphs and excessive particles. Absolutely no text, no letters, no numbers, no watermark, no people, no faces, no company logos, no branded GPU hardware, no arrows. Only the concept artwork, full bleed, no border.
```

### 3.2 B 版：imagegen API

- 工作流：用户批准 API 对照方案，使用 `codex-imagegen-preview` 受控 CLI 的 `images` 模式，单次生成。
- 请求模型：`gpt-image-2`；本机配置的 API relay provider 为 `verdantflare`。A / B 是不同调用入口及请求模型的候选，不能据此推断实际底层模型彼此独立。
- 请求规格：3840 × 2160、high；实际规格见第 2 节。
- 原图已完成本地解码、尺寸和视觉检查。

实际提交提示词：

```text
Use case: stylized-concept
Asset type: landscape 16:9 cover background for the Vela technical whitepaper, about a multi-GPU streaming engine for real-world video super-resolution.
Create an exceptionally refined editorial technology cover illustration. Deep midnight navy background, near-black at the left, restrained teal light with a few warm copper accents. On the right, a sculptural stream of translucent cinematic image planes and precisely arranged parallel compute modules: a subtle transition from soft coarse pixel fragments to clean sharply resolved optical glass sheets. Convey video restoration, continuity, and parallel processing through physical material, light and composition. The modules should feel like finely crafted industrial instruments rather than branded computer parts. Oblique perspective, beautiful controlled edge lighting, realistic frosted glass and brushed dark metal, exquisite fine detail, quiet premium print-design aesthetic.
Composition: keep the LEFT 45 percent nearly empty and dark for later typography, all intricate visual activity on the RIGHT 55 percent. Maintain generous margins, a calm upper-left corner and a quiet bottom strip. A coherent single visual composition, not a collage or a dashboard. Wide cinematic 16:9 image at high resolution.
Palette: midnight navy #10263d, near black #071321, restrained teal #087f8c and luminous pale cyan, very sparse copper-gold accents.
Constraints: absolutely NO text, letters, numbers, labels, typography, watermark or logos; no people, faces, robots, brains, generic network globes, circuitry wallpaper, neon purple, UI widgets, literal architecture diagrams or arrows. This is an abstract editorial cover, not a benchmark or restoration example. Preserve left negative space for a crisp title added separately.
```

## 4. 校验

SVG 已通过 XML 解析、字体宽度检查与 Chrome 实际渲染检查；Markdown 内嵌图片及本地相对链接已检查。封面已检查文字、边距和可读性。

以下为最终交付文件的 SHA-256：

- `vela-overall-architecture.svg`：`567808f8ef3e4ea5e86fc60e566f18dabaeaed96a03cebe83a449d7389dd4f16`。
- `vela-technical-architecture.svg`：`bad7b83112adb994cd1d094860db7a169e9d30027736c350af2767881f5b36bd`。
- `vela-whitepaper-cover-vf.png`：`0f2d193cabc00f95c496e8250b38c8e704f5c2e02b2e11387fb9ee49d45411f2`。
- `vela-whitepaper-cover-imagegen.png`：`0953691cbce0a3aa4871909fd8c7a0d9d4f0fa2f9cba69eb19fc8964c6634134`。
- `vela-whitepaper-cover-vf-art.png`：`4632b440f6a22ed4da064f99a3421a5bcfa29e398853c84060dc78e815a20293`。
- `vela-whitepaper-cover-imagegen-art.png`：`70ab04bcf0211f22d8c68bcba06f99a5171a6f548136c434fdf4346a95e29044`。
