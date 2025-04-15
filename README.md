# XSTech OpenAI 兼容API

这个项目提供了一个将XSTech API转换为OpenAI API格式的服务，支持cookie池管理和会话上下文处理。

## 特性

- OpenAI API兼容接口
- Cookie池管理，支持多账号负载均衡
- 自动处理会话上下文
- 支持图片上传
- 支持流式输出
- 定期清理长时间不活跃的会话

## 安装

### 前提条件

- Python 3.8+
- pip

### 安装步骤

1. 克隆仓库

```bash
git clone https://github.com/1hei1/xt2api.git
cd xt2api
```

2. 安装依赖

```bash
pip install -r requirements.txt
```

## 配置

在运行服务前，需要配置XSTech的cookie信息。可以通过以下步骤:

1. 创建配置文件 `config.json`:

```json
{
  "cookies": [
    {
      "authorization": "你的XSTech authorization令牌"
    },
    {
      "authorization": "另一个XSTech authorization令牌"
    }
  ]
}
```

2. 修改 `xstech_openai_api.py` 的 `startup_event` 函数，从配置文件加载cookie:

```python
@app.on_event("startup")
async def startup_event():
    """启动时初始化"""
    # 从配置文件加载cookie
    with open("config.json", "r") as f:
        config = json.load(f)
        for cookie_data in config["cookies"]:
            cookie_pool.add_cookie(cookie_data)
    
    # 初始化模型映射
    await xstech_client.fetch_models()
    
    # 启动会话清理任务
    await session_manager.start_cleanup_task()
```

## 运行

运行API服务:

```bash
uvicorn xstech_openai_api:app --host 0.0.0.0 --port 8000
```

或者直接运行Python脚本:

```bash
python xstech_openai_api.py
```

## API使用

### 获取可用模型

```
GET /v1/models
```

返回XSTech上所有积分为1的可用模型。

### 聊天补全

```
POST /v1/chat/completions
```

请求体(OpenAI兼容格式):

```json
{
  "model": "xstech-claude-3-7-sonnet-20250219-xs",
  "messages": [
    {
      "role": "user",
      "content": "你好，请介绍一下自己"
    }
  ],
  "stream": false
}
```

对于图片输入，使用OpenAI兼容的多模态格式:

```json
{
  "model": "xstech-claude-3-7-sonnet-20250219-xs",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "这张图片是什么?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,/9j/4AAQSk..."
          }
        }
      ]
    }
  ],
  "stream": false
}
```

## 健康检查

```
GET /health
```

返回API服务状态和cookie池大小。

## 限制和注意事项

- 目前仅支持积分为1的XSTech模型
- 会话在1小时不活跃后自动清理
- 图片支持仅限于base64编码的图片
- 需要有效的XSTech authorization令牌

## 排错

如果遇到问题，请检查日志输出，服务会记录详细的操作和错误信息。

## 许可证

MIT 