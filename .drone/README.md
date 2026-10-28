# vela

## CI/CD

```powershell
# 触发 release 构建 (推送运行时镜像至阿里云)
git checkout release ;`
  git merge main --ff-only ;`
  git push origin release ;`
  git checkout main

# 触发 build 构建 (推送基础 Builder 镜像至 GHCR)
git checkout build ;`
  git merge main --ff-only ;`
  git push origin build ;`
  git checkout main
```

## Local Debugging

```powershell
# 1. 本地测试构建 builder 环境镜像
docker build -f .drone/dockerfile.build -t ghcr.io/mengkzhaoyun/vela:v0.1.0-build --build-arg BASE=nvidia/cuda:13.0.3-devel-ubuntu24.04 .

# 2. 本地测试运行镜像构建
docker build -f .drone/dockerfile -t registry.cn-qingdao.aliyuncs.com/wod/cuda:13.0.3-vela-v0.1.0 .
```

## Production Deployment (RTX 4090 / RTX 5090 多卡统一镜像)

Vela 容器启动后默认探测全部可用 GPU（`VELA_GPUS=all`），采用 **GOP 关键帧动态抢占队列（Work-Stealing）** 对 `/data` 目录下的低分辨率视频（`< 1080p`）执行流式超分：

```text
FFmpeg 时空预去噪 (hqdn3d) -> 多卡 GOP 动态抢占队列 -> Spandrel PLKSR 4x 超分 -> GPU 抗锯齿 1080p 下采样 + GPU CAS 自适应锐化 -> NVENC 5M 硬件编码 -> 无损 Concat & 音轨 Remux
```

### 1. 多卡全速生产部署（默认推荐 `realplksr`，绑定 `/data/jasna/ais`）

```bash
docker pull registry.cn-qingdao.aliyuncs.com/wod/cuda:13.0.3-vela-v0.1.0 && \
docker rm -f vela || true && \
docker run --name vela \
  -d --rm \
  --gpus all \
  -v /data/jasna/model_weights:/app/vela/model_weights \
  -v /data/jasna/ais:/data \
  registry.cn-qingdao.aliyuncs.com/wod/cuda:13.0.3-vela-v0.1.0
```

### 2. 指定单文件与指定显卡超分（支持 `.mkv` / `.mp4` / `.avi` / `.wmv` 等格式）

```bash
docker run --rm --gpus all \
  -v /data/jasna/model_weights:/app/vela/model_weights \
  -v /data/jasna/ais:/data \
  registry.cn-qingdao.aliyuncs.com/wod/cuda:13.0.3-vela-v0.1.0 \
  --input /data/input_480p.mkv \
  --output /data/output_1080p.mp4 \
  --gpus 0,1 \
  --model realplksr
```

### 模型手动下载 (离线/网络受限环境)

```bash
mkdir -p /data/jasna/model_weights
cd /data/jasna/model_weights

# 1. RealPLKSR 写实去压缩块 4x 超分模型 (默认首选: 4xNomosWebPhoto_RealPLKSR.safetensors / .pth)
curl -L -o 4xNomosWebPhoto_RealPLKSR.safetensors https://hf-mirror.com/Phhofm/4xNomosWebPhoto_RealPLKSR/resolve/main/4xNomosWebPhoto_RealPLKSR.safetensors
ln -sfn 4xNomosWebPhoto_RealPLKSR.safetensors 4xNomosWebPhoto_RealPLKSR.pth

# 2. ClearRealityV1 五官与边缘锐化 4x 超分模型 (可选: 4x-ClearRealityV1.pth)
curl -L -O https://hf-mirror.com/Kim2091/ClearRealityV1/resolve/main/4x-ClearRealityV1.pth
```
