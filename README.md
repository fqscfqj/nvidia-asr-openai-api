# Canary ASR Docker API

基于 NVIDIA Canary-1B-v2 模型的 OpenAI Whisper 兼容语音识别 API 服务。

## 功能特性

- 🚀 **懒加载 (Lazy Loading)**: 启动时不加载模型，首次请求时才加载到 GPU，节省资源
- ⏱️ **自动卸载 (Auto-Unload)**: 模型闲置超时后自动释放 GPU 显存
- 🔒 **线程安全**: 确保并发请求安全，模型使用中不会被卸载
- 🎯 **高精度识别**: 基于 10 亿参数 Canary 模型，支持 25 种欧洲语言
- 📝 **多格式输出**: 支持 text/json/srt/vtt/verbose_json 格式
- 🔌 **OpenAI 兼容**: 完全兼容 OpenAI Whisper API 接口

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
cd canary_asr_docker
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
curl http://localhost:8000/health

# 查看模型状态
curl http://localhost:8000/status

# 转录音频
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@your_audio.wav \
  -F language=en \
  -F response_format=json
```

## API 文档

### 音频转录

**端点**: `POST /v1/audio/transcriptions`

**请求参数**:

| 参数 | 类型 | 必填 | 说明 |
|-----|------|-----|------|
| file | file | 是 | 音频文件 (支持 wav, flac, mp3, m4a 等) |
| model | string | 否 | 模型名称 (兼容参数) |
| language | string | 否 | 语言代码，如 'en', 'de' |
| response_format | string | 否 | 响应格式: text, json, srt, vtt, verbose_json |

**Python 示例**:

```python
import requests

url = "http://localhost:8000/v1/audio/transcriptions"

# 上传文件并转录
with open("audio.wav", "rb") as f:
    response = requests.post(
        url,
        files={"file": f},
        data={
            "language": "en",
            "response_format": "json"
        }
    )

print(response.json())
# 输出: {"text": "转录的文本内容..."}
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
    "http://localhost:8000/v1/audio/translations",
    files={"file": open("german_audio.wav", "rb")},
    data={"response_format": "json"}
)

print(response.json())
# 输出: {"text": "English translation..."}
```

### 模型管理

**预加载模型**:
```bash
curl -X POST http://localhost:8000/model/load
```

**卸载模型**:
```bash
curl -X POST http://localhost:8000/model/unload
```

**查看状态**:
```bash
curl http://localhost:8000/status
```

## 环境变量配置

| 变量名 | 默认值 | 说明 |
|-------|-------|------|
| MODEL_PATH | /data/model | 模型存储路径 |
| MODEL_NAME | nvidia/canary-1b-v2 | HuggingFace 模型名称 |
| MODEL_TIMEOUT_SEC | 300 | 模型闲置超时时间 (秒) |
| USE_FP16 | true | 是否使用 FP16 半精度推理 |
| API_PORT | 8000 | API 服务端口 |
| LOG_LEVEL | INFO | 日志级别 |

## 项目结构

```
canary_asr_docker/
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

本项目遵循 MIT 许可证。NVIDIA Canary 模型遵循 CC-BY-4.0 许可证。

## 致谢

- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)
- [NVIDIA Canary-1B-v2](https://huggingface.co/nvidia/canary-1b-v2)
- [FastAPI](https://fastapi.tiangolo.com/)
