# NVIDIA ASR to OpenAI API

基于 NVIDIA canary-1b-v2 和 parakeet-tdt-0.6b-v3 模型的 OpenAI Whisper 兼容语音识别 API 服务。

## 功能特性

- 🚀 **懒加载 (Lazy Loading)**: 启动时不加载模型，首次请求时才加载到 GPU，节省资源
- ⏱️ **自动卸载 (Auto-Unload)**: 模型闲置超时后自动释放 GPU 显存
- 🔒 **线程安全**: 确保并发请求安全，模型使用中不会被卸载
- 🎯 **多模型支持**: 支持 canary-1b-v2 和 parakeet-tdt-0.6b-v3，可通过环境变量或 API 参数选择
- 📝 **多格式输出**: 支持 text/json/srt/vtt/verbose_json 格式
- 🔌 **OpenAI 兼容**: 完全兼容 OpenAI Whisper API 接口

## 支持的模型

| 模型名称 | 模型 ID | 参数量 | 支持语言 | 特点 |
|---------|---------|-------|---------|------|
| NVIDIA Canary 1B v2 | canary-1b-v2 | 10 亿 | 25 种欧洲语言 | 高精度识别，支持多语言 ASR 和 AST |
| NVIDIA Parakeet TDT 0.6B v3 | parakeet-tdt-0.6b-v3 | 6 亿 | 主要英语 | 轻量级快速模型 |

### 配置启用的模型

在 `docker-compose.yml` 中设置 `ENABLED_MODELS` 环境变量：

```yaml
environment:
  # 仅启用 Canary 模型
  - ENABLED_MODELS=canary-1b-v2
  
  # 仅启用 Parakeet 模型
  # - ENABLED_MODELS=parakeet-tdt-0.6b-v3
  
  # 同时启用两个模型 (逗号分隔)
  # - ENABLED_MODELS=canary-1b-v2,parakeet-tdt-0.6b-v3
```

## 支持的语言

| 语言代码 | 语言名称 | 语言代码 | 语言名称 |
|---------|---------|---------|---------|
| en | 英语 | de | 德语 |
| fr | 法语 | es | 西班牙语 |
| it | 意大利语 | pt | 葡萄牙语 |
| nl | 荷兰语 | pl | 波兰语 |
| ru | 俄语 | uk | 乌克兰语 |
| cs | 捷克语 | sk | 斯洛伐克语 |
| bg | 保加利亚语 | hr | 克罗地亚语 |
| da | 丹麦语 | fi | 芬兰语 |
| el | 希腊语 | hu | 匈牙利语 |
| ro | 罗马尼亚语 | sv | 瑞典语 |
| et | 爱沙尼亚语 | lv | 拉脱维亚语 |
| lt | 立陶宛语 | sl | 斯洛文尼亚语 |
| mt | 马耳他语 | | |

## 快速开始

### 前置要求

- Docker 和 Docker Compose
- NVIDIA GPU (建议至少 6GB 显存)
- NVIDIA Container Toolkit

### 1. 克隆项目

```bash
git clone <repository-url>
cd nvidia-asr-openai-api
```

### 2. 创建模型目录

```bash
mkdir -p models
```

### 3. 启动服务

```bash
docker-compose up -d
```

首次启动时，如果本地没有模型文件，会自动从 HuggingFace 下载 (约需 10-20 分钟)。

### 4. 测试 API

```bash
# 健康检查
curl http://localhost:8909/health

# 查看模型状态
curl http://localhost:8909/status

# 转录音频
curl -X POST http://localhost:8909/v1/audio/transcriptions \
  -F file=@your_audio.wav \
  -F language=en \
  -F response_format=json
```

## API 文档

API 文档可通过访问 `http://localhost:8909/docs` 查看。

### 获取模型列表

**端点**: `GET /v1/models`

获取当前启用的所有模型列表，兼容 OpenAI API 格式。

**请求示例**:

```bash
# 无需认证
curl http://localhost:8909/v1/models

# 如果启用了 API Key 验证
curl -H "Authorization: Bearer your-api-key" http://localhost:8909/v1/models
```

**响应示例**:

```json
{
  "object": "list",
  "data": [
    {
      "id": "canary-1b-v2",
      "object": "model",
      "created": 1699000000,
      "owned_by": "nvidia"
    },
    {
      "id": "parakeet-tdt-0.6b-v3",
      "object": "model",
      "created": 1699000000,
      "owned_by": "nvidia"
    }
  ]
}
```

### 音频转录

**端点**: `POST /v1/audio/transcriptions`

**请求参数**:

| 参数 | 类型 | 必填 | 说明 |
|-----|------|-----|------|
| file | file | 是 | 音频文件 (支持 wav, flac, mp3, m4a 等) |
| model | string | 否 | 模型名称: canary-1b-v2 或 parakeet-tdt-0.6b-v3 (默认: canary-1b-v2) |
| language | string | 否 | 语言代码，如 'en', 'de' |
| response_format | string | 否 | 响应格式: text, json, srt, vtt, verbose_json |

**Python 示例**:

```python
import requests

url = "http://localhost:8909/v1/audio/transcriptions"

# 基本使用 (无 API Key)
with open("audio.wav", "rb") as f:
    response = requests.post(
        url,
        files={"file": f},
        data={
            "model": "canary-1b-v2",
            "language": "en",
            "response_format": "json"
        }
    )

print(response.json())
# 输出: {"text": "转录的文本内容..."}

# 使用 API Key 认证
headers = {"Authorization": "Bearer your-api-key"}
with open("audio.wav", "rb") as f:
    response = requests.post(
        url,
        headers=headers,
        files={"file": f},
        data={
            "model": "canary-1b-v2",
            "language": "en",
            "response_format": "json"
        }
    )

# 使用 Parakeet 模型转录
with open("audio.wav", "rb") as f:
    response = requests.post(
        url,
        files={"file": f},
        data={
            "model": "parakeet-tdt-0.6b-v3",
            "language": "en",
            "response_format": "json"
        }
    )
```

**获取 SRT 字幕**:

```python
response = requests.post(
    url,
    files={"file": open("audio.wav", "rb")},
    data={"response_format": "srt"}
)

print(response.text)
# 输出:
# 1
# 00:00:00,000 --> 00:00:02,500
# 第一段字幕
#
# 2
# 00:00:02,500 --> 00:00:05,000
# 第二段字幕
```

### 音频翻译

**端点**: `POST /v1/audio/translations`

将任意支持的语言翻译为英语。

```python
response = requests.post(
    "http://localhost:8909/v1/audio/translations",
    files={"file": open("german_audio.wav", "rb")},
    data={"response_format": "json"}
)

print(response.json())
# 输出: {"text": "English translation..."}
```

### 模型管理

**预加载模型**:
```bash
curl -X POST http://localhost:8909/model/load
```

**卸载模型**:
```bash
curl -X POST http://localhost:8909/model/unload
```

**查看状态**:
```bash
curl http://localhost:8909/status
```

## 环境变量配置

| 变量名 | 默认值 | 说明 |
|-------|-------|------|
| MODEL_PATH | /data/model | 模型存储路径 |
| MODEL_NAME | nvidia/canary-1b-v2 | HuggingFace 模型名称 |
| MODEL_TIMEOUT_SEC | 300 | 模型闲置超时时间 (秒) |
| ENABLED_MODELS | canary-1b-v2 | 启用的模型列表 (逗号分隔) |
| USE_FP16 | true | 是否使用 FP16 半精度推理 |
| API_PORT | 8909 | API 服务端口 |
| LOG_LEVEL | INFO | 日志级别 |
| API_KEY | (空) | API Key 认证密钥 (可选) |

### API Key 配置

为了保护 API 安全，可以设置 API Key 进行身份验证：

1. **设置 API Key**:

在 `docker-compose.yml` 中设置 `API_KEY` 环境变量：

```yaml
environment:
  - API_KEY=your-secret-api-key-here
```

或使用 `.env` 文件：

```bash
# 复制示例文件
cp .env.example .env

# 编辑 .env 文件，取消注释并设置 API_KEY
# API_KEY=your-secret-api-key-here
```

2. **使用 API Key 请求**:

```bash
# 使用 curl
curl -H "Authorization: Bearer your-secret-api-key-here" \
  http://localhost:8909/v1/models

# 使用 Python
import requests

headers = {"Authorization": "Bearer your-secret-api-key-here"}
response = requests.get("http://localhost:8909/v1/models", headers=headers)
```

3. **禁用 API Key**:

如果不需要认证，只需注释掉或删除 `API_KEY` 环境变量即可。

**注意**: 
- 如果设置了 `API_KEY`，所有 API 端点（除了 `/health` 和 `/`）都需要提供有效的 API Key
- `/health` 端点不需要认证，用于健康检查
- 请妥善保管 API Key，不要泄露给他人

## 项目结构

```
nvidia-asr-openai-api/
├── docker-compose.yml    # Docker Compose 配置
├── Dockerfile            # Docker 镜像构建文件
├── requirements.txt      # Python 依赖
├── README.md            # 项目文档
├── models/              # 模型存储目录 (挂载卷)
└── src/
    ├── __init__.py      # 包初始化
    ├── main.py          # FastAPI 主应用
    ├── model_manager.py # 模型生命周期管理
    ├── engine.py        # 推理引擎
    └── utils.py         # 工具函数
```

## 性能优化建议

1. **预加载模型**: 生产环境可在启动后调用 `/model/load` 预热模型
2. **调整超时时间**: 根据使用频率调整 `MODEL_TIMEOUT_SEC`
3. **GPU 显存**: Canary 模型约需 4-6GB 显存，FP16 模式可减少约 50%
4. **批量处理**: 对于大量文件，考虑串行处理避免显存溢出

## 常见问题

### Q: 首次请求很慢？
A: 首次请求会触发模型加载，需要 30-60 秒。可以提前调用 `/model/load` 预热。

### Q: 显存不足？
A: 确保启用了 FP16 模式 (`USE_FP16=true`)，并使用至少 6GB 显存的 GPU。

### Q: 模型下载失败？
A: 检查网络连接，或手动下载模型文件放到 `models/` 目录。

## 许可证

本项目遵循 MIT 许可证。NVIDIA ASR 模型遵循 CC-BY-4.0 许可证。

## 致谢

- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)
- [NVIDIA canary-1b-v2](https://huggingface.co/nvidia/canary-1b-v2)
- [NVIDIA parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- [FastAPI](https://fastapi.tiangolo.com/)
