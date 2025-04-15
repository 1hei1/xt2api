import base64
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Union
import asyncio
import logging
from datetime import datetime, timedelta
import random

import aiohttp
from fastapi import FastAPI, Depends, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API 模型定义
class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = False
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    
class ImageData(BaseModel):
    url: Optional[str] = None
    data: Optional[str] = None
    
class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class CompletionMessage(BaseModel):
    role: str
    content: str

class CompletionChoice(BaseModel):
    index: int
    message: CompletionMessage
    finish_reason: str

class ChatCompletion(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: UsageInfo

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "xstech"

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]

# Cookie池管理
class CookiePool:
    def __init__(self):
        self.cookies: List[Dict[str, Any]] = []
        self.in_use: Dict[str, bool] = {}
        self.cookie_stats: Dict[str, Dict[str, Any]] = {}
        
    def add_cookie(self, cookie_data: Dict[str, str]):
        """添加cookie到池中"""
        cookie_id = str(uuid.uuid4())
        self.cookies.append({
            "id": cookie_id,
            "data": cookie_data,
            "last_used": 0
        })
        self.in_use[cookie_id] = False
        self.cookie_stats[cookie_id] = {
            "requests": 0,
            "errors": 0,
            "last_usage_time": None
        }
        return cookie_id
        
    def get_cookie(self) -> Optional[Dict[str, Any]]:
        """获取一个可用的cookie"""
        available_cookies = [c for c in self.cookies if not self.in_use.get(c["id"], False)]
        if not available_cookies:
            return None
            
        # 选择最长时间未使用的cookie
        selected_cookie = min(available_cookies, key=lambda x: x["last_used"])
        self.in_use[selected_cookie["id"]] = True
        selected_cookie["last_used"] = time.time()
        self.cookie_stats[selected_cookie["id"]]["last_usage_time"] = datetime.now()
        self.cookie_stats[selected_cookie["id"]]["requests"] += 1
        return selected_cookie
        
    def release_cookie(self, cookie_id: str, error: bool = False):
        """释放cookie，标记为可用"""
        if cookie_id in self.in_use:
            self.in_use[cookie_id] = False
            if error and cookie_id in self.cookie_stats:
                self.cookie_stats[cookie_id]["errors"] += 1

# 会话管理
class SessionManager:
    def __init__(self, cookie_pool: CookiePool):
        self.cookie_pool = cookie_pool
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.conversation_to_session: Dict[str, str] = {}
        self.last_activity: Dict[str, float] = {}
        self.session_locks: Dict[str, asyncio.Lock] = {}
        self.cleanup_task = None
        
    async def start_cleanup_task(self):
        """启动定期清理任务"""
        if self.cleanup_task is None:
            self.cleanup_task = asyncio.create_task(self._cleanup_sessions())
        
    async def _cleanup_sessions(self):
        """定期清理长时间不活跃的会话"""
        while True:
            try:
                now = time.time()
                to_remove = []
                
                for session_id, last_active in self.last_activity.items():
                    if now - last_active > 3600:  # 1小时不活跃的会话
                        to_remove.append(session_id)
                
                for session_id in to_remove:
                    await self.remove_session(session_id)
                    
                # 从conversation_to_session中清理已删除的session
                stale_conversations = [conv_id for conv_id, session_id in self.conversation_to_session.items() 
                                     if session_id not in self.sessions]
                for conv_id in stale_conversations:
                    del self.conversation_to_session[conv_id]
                    
            except Exception as e:
                logger.error(f"Error in session cleanup: {e}")
                
            await asyncio.sleep(300)  # 每5分钟检查一次
    
    async def get_session_for_conversation(self, conversation_id: str, model: str) -> str:
        """为对话获取或创建一个会话"""
        if conversation_id in self.conversation_to_session:
            session_id = self.conversation_to_session[conversation_id]
            if session_id in self.sessions:
                self.last_activity[session_id] = time.time()
                return session_id
        
        # 需要创建新会话
        return await self.create_session(conversation_id, model)
    
    async def create_session(self, conversation_id: str, model: str) -> str:
        """创建新的xstech会话"""
        cookie = self.cookie_pool.get_cookie()
        if not cookie:
            raise HTTPException(status_code=503, detail="No available cookies")
        
        try:
            headers = {
                "accept": "application/json, text/plain, */*",
                "authorization": cookie["data"]["authorization"],
                "x-app-version": "2.1.1",
                "content-type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://xstech.one/api/chat/session",
                    headers=headers
                ) as response:
                    if response.status != 200:
                        self.cookie_pool.release_cookie(cookie["id"], error=True)
                        raise HTTPException(status_code=response.status, detail="Failed to create xstech session")
                    
                    result = await response.json()
                    if result.get("code") != 0:
                        self.cookie_pool.release_cookie(cookie["id"], error=True)
                        raise HTTPException(status_code=400, detail=f"xstech API error: {result.get('msg', 'Unknown error')}")
                    
                    session_id = str(result["data"]["id"])
                    self.sessions[session_id] = {
                        "cookie_id": cookie["id"],
                        "model": model,
                        "xstech_session_id": session_id,
                        "created_at": datetime.now(),
                        "messages": []
                    }
                    self.conversation_to_session[conversation_id] = session_id
                    self.last_activity[session_id] = time.time()
                    self.session_locks[session_id] = asyncio.Lock()
                    self.cookie_pool.release_cookie(cookie["id"])
                    return session_id
                    
        except Exception as e:
            self.cookie_pool.release_cookie(cookie["id"], error=True)
            logger.error(f"Error creating session: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")
    
    async def remove_session(self, session_id: str):
        """删除会话"""
        if session_id in self.sessions:
            # 从对话映射中删除
            for conv_id, sess_id in list(self.conversation_to_session.items()):
                if sess_id == session_id:
                    del self.conversation_to_session[conv_id]
            
            # 删除会话数据
            del self.sessions[session_id]
            if session_id in self.last_activity:
                del self.last_activity[session_id]
            if session_id in self.session_locks:
                del self.session_locks[session_id]
    
    async def get_session_lock(self, session_id: str) -> asyncio.Lock:
        """获取会话锁，确保会话的并发安全"""
        if session_id not in self.session_locks:
            self.session_locks[session_id] = asyncio.Lock()
        return self.session_locks[session_id]

# xstech API 客户端
class XSTechClient:
    def __init__(self, cookie_pool: CookiePool, session_manager: SessionManager):
        self.cookie_pool = cookie_pool
        self.session_manager = session_manager
        self.model_map = {}  # 将在初始化时填充
        self.reverse_model_map = {}  # 反向映射
        
    async def fetch_models(self):
        """获取xstech支持的模型列表"""
        cookie = self.cookie_pool.get_cookie()
        if not cookie:
            raise HTTPException(status_code=503, detail="No available cookies")
            
        try:
            headers = {
                "accept": "application/json, text/plain, */*",
                "authorization": cookie["data"]["authorization"],
                "x-app-version": "2.1.1"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://xstech.one/api/chat/tmpl",
                    headers=headers
                ) as response:
                    if response.status != 200:
                        self.cookie_pool.release_cookie(cookie["id"], error=True)
                        raise HTTPException(status_code=response.status, detail="Failed to fetch models")
                    
                    result = await response.json()
                    if result.get("code") != 0:
                        self.cookie_pool.release_cookie(cookie["id"], error=True)
                        raise HTTPException(status_code=400, detail=f"xstech API error: {result.get('msg', 'Unknown error')}")
                    
                    models = []
                    # 只保留积分为1的模型
                    for model in result["data"]["models"]:
                        integral = model.get("attr", {}).get("integral", "")
                        if integral == "1积分":
                            xstech_id = model["value"]
                            openai_compatible_id = f"xstech-{xstech_id}"
                            self.model_map[openai_compatible_id] = xstech_id
                            self.reverse_model_map[xstech_id] = openai_compatible_id
                            models.append({
                                "id": openai_compatible_id,
                                "created": int(time.time()),
                                "name": model["label"]
                            })
                    
                    self.cookie_pool.release_cookie(cookie["id"])
                    return models
                    
        except Exception as e:
            if cookie:
                self.cookie_pool.release_cookie(cookie["id"], error=True)
            logger.error(f"Error fetching models: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to fetch models: {str(e)}")
    
    async def send_chat_completion(self, session_id: str, text: str, files: List[Dict[str, str]] = None) -> tuple[aiohttp.ClientSession, aiohttp.ClientResponse]:
        """发送聊天补全请求到xstech API，并返回session和response"""
        if session_id not in self.session_manager.sessions:
            raise HTTPException(status_code=404, detail="Session not found")
            
        session_data = self.session_manager.sessions[session_id]
        cookie_id = session_data["cookie_id"]
        cookie = next((c for c in self.cookie_pool.cookies if c["id"] == cookie_id), None)
        
        if not cookie:
            # 如果找不到cookie，可能需要释放会话或采取其他措施
            logger.error(f"Cookie {cookie_id} not found for session {session_id}")
            raise HTTPException(status_code=503, detail="Session cookie not available or invalid")
            
        # 标记cookie正在使用
        self.cookie_pool.in_use[cookie_id] = True
        xstech_session_id = session_data["xstech_session_id"]
        
        http_session = None # Initialize http_session to None
        try:
            headers = {
                "accept": "text/event-stream",
                "authorization": cookie["data"]["authorization"],
                "content-type": "application/json",
                "x-app-version": "2.1.1",
                # 添加一些常见的浏览器头，可能有助于防止被服务器拒绝
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Referer": "https://xstech.one/chat",
                "Origin": "https://xstech.one"
            }
            
            payload = {
                "text": text,
                "sessionId": int(xstech_session_id),
                "files": files or []
            }
            
            # 手动创建 aiohttp session
            http_session = aiohttp.ClientSession()
            
            response = await http_session.post(
                "https://xstech.one/api/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300) # 增加超时时间
            )
            
            # 检查状态码
            if response.status != 200:
                error_detail = f"XSTech API request failed with status {response.status}. Response: {await response.text()}"
                logger.error(error_detail)
                await http_session.close() # 确保关闭session
                self.cookie_pool.release_cookie(cookie_id, error=True)
                raise HTTPException(status_code=response.status, detail="XSTech API request failed")
                
            # 更新会话活动时间
            self.session_manager.last_activity[session_id] = time.time()
            
            # 返回session和response，由调用者负责关闭session
            return http_session, response
                
        except aiohttp.ClientConnectionError as e:
            logger.error(f"Connection error during chat completion for session {session_id}: {e}")
            if http_session and not http_session.closed:
                await http_session.close()
            self.cookie_pool.release_cookie(cookie_id, error=True)
            raise HTTPException(status_code=503, detail=f"Service connection error: {e}")
        except asyncio.TimeoutError as e:
            logger.error(f"Timeout error during chat completion for session {session_id}: {e}")
            if http_session and not http_session.closed:
                await http_session.close()
            self.cookie_pool.release_cookie(cookie_id, error=True)
            raise HTTPException(status_code=504, detail=f"Request timed out: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in chat completion for session {session_id}: {e}", exc_info=True)
            if http_session and not http_session.closed:
                await http_session.close()
            # 只有在确认是cookie问题时才标记错误，否则可能只是临时网络问题
            # self.cookie_pool.release_cookie(cookie_id, error=True) 
            self.cookie_pool.release_cookie(cookie_id) # 暂时先不标记错误
            raise HTTPException(status_code=500, detail=f"Chat completion error: {str(e)}")

    async def update_session_model(self, session_id: str, new_model: str) -> bool:
        """更新XSTech会话使用的模型"""
        if session_id not in self.session_manager.sessions:
            raise HTTPException(status_code=404, detail="会话不存在")
        
        session_data = self.session_manager.sessions[session_id]
        cookie_id = session_data["cookie_id"]
        cookie = next((c for c in self.cookie_pool.cookies if c["id"] == cookie_id), None)
        
        if not cookie:
            raise HTTPException(status_code=503, detail="会话Cookie不可用")
        
        xstech_session_id = session_data["xstech_session_id"]
        
        try:
            headers = {
                "accept": "application/json, text/plain, */*",
                "authorization": cookie["data"]["authorization"],
                "content-type": "application/json",
                "x-app-version": "2.1.1"
            }
            
            # 构建请求体，保留现有会话设置但更新模型
            body = {
                "id": int(xstech_session_id),
                "model": new_model,
                # 可以添加更多默认参数
                "contextCount": 10,
                "temperature": 0,
                "presencePenalty": 0,
                "frequencyPenalty": 0,
                "prompt": "",
                "plugins": None,
                "localPlugins": None,
                "useAppId": 0
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.put(
                    f"https://xstech.one/api/chat/session/{xstech_session_id}",
                    headers=headers,
                    json=body
                ) as response:
                    if response.status != 200:
                        return False
                    
                    result = await response.json()
                    if result.get("code") != 0:
                        return False
                    
                    # 更新本地会话记录
                    session_data["model"] = new_model
                    return True
                
        except Exception as e:
            logger.error(f"更新会话模型失败: {e}")
            return False

# 处理图片数据
def process_image_content(content: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """处理OpenAI格式的图片内容，转换为xstech格式"""
    files = []
    for item in content:
        if item.get("type") == "image_url":
            image_url = item.get("image_url", {})
            if isinstance(image_url, dict) and "url" in image_url:
                # 处理base64图片
                if image_url["url"].startswith("data:image/"):
                    parts = image_url["url"].split(",", 1)
                    if len(parts) == 2:
                        image_data = parts[1]
                        name = f"image_{len(files)}.png"
                        files.append({
                            "name": name,
                            "data": image_url["url"]
                        })
            elif isinstance(image_url, str) and image_url.startswith("data:image/"):
                parts = image_url.split(",", 1)
                if len(parts) == 2:
                    image_data = parts[1]
                    name = f"image_{len(files)}.png"
                    files.append({
                        "name": name,
                        "data": image_url
                    })
    return files

# 创建API应用
app = FastAPI(title="xstech OpenAI Compatible API")

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化组件
cookie_pool = CookiePool()
session_manager = SessionManager(cookie_pool)
xstech_client = XSTechClient(cookie_pool, session_manager)

# 启动时加载cookie
@app.on_event("startup")
async def startup_event():
    """启动时初始化"""
    # 添加cookie示例，实际应用中可从配置文件或环境变量加载
    cookie_pool.add_cookie({
        "authorization": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOjIwNzUyLCJzaWduIjoiYTQ1NjAwNjNiNTQ1ZGU3NTFlYjVhZjNjNGZmNWM2NTkiLCJyb2xlIjoidXNlciIsImV4cCI6MTc0NTI4OTczOSwibmJmIjoxNzQ0Njg0OTM5LCJpYXQiOjE3NDQ2ODQ5Mzl9.friooqRWpNRQhTieNQfVxoALX34TxWli5X8dIFJc06I",
    })
    
    # 初始化模型映射
    await xstech_client.fetch_models()
    
    # 启动会话清理任务
    await session_manager.start_cleanup_task()

# 获取模型列表
@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """获取所有支持的模型"""
    try:
        models = await xstech_client.fetch_models()
        return {
            "object": "list",
            "data": [
                {
                    "id": model["id"],
                    "object": "model",
                    "created": model["created"],
                    "owned_by": "xstech"
                }
                for model in models
            ]
        }
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

# 生成会话ID
def generate_conversation_id(messages: List[ChatMessage]) -> str:
    """根据消息生成会话ID"""
    # 使用前几条消息的内容哈希作为会话ID
    content_hash = ""
    for i, msg in enumerate(messages[:3]):  # 使用前3条消息
        if isinstance(msg.content, str):
            content_hash += msg.content
        elif isinstance(msg.content, list):
            for item in msg.content:
                if item.get("type") == "text":
                    content_hash += item.get("text", "")
    
    return str(hash(content_hash))

# 聊天补全API
@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, background_tasks: BackgroundTasks):
    """处理聊天补全请求"""
    http_session = None # Initialize http_session outside try
    session_id = None   # Initialize session_id outside try
    cookie_id = None    # Initialize cookie_id outside try
    
    try:
        # 检查模型是否支持
        if not request.model.startswith("xstech-"):
            xstech_model = xstech_client.reverse_model_map.get(request.model) # 尝试反向查找
            if not xstech_model:
                 # 如果反向查找不到，并且不是 xstech- 前缀，则报错
                 raise HTTPException(status_code=400, detail=f"Model {request.model} not supported or invalid format. Use /v1/models to list supported models.")
        else:
             # 如果是 xstech- 前缀，正常查找
            xstech_model_mapped = xstech_client.model_map.get(request.model)
            if not xstech_model_mapped:
                raise HTTPException(status_code=400, detail=f"Model {request.model} not found. Use /v1/models to list supported models.")
            xstech_model = xstech_model_mapped # Use the mapped value

        # 从消息生成会话ID
        conversation_id = generate_conversation_id(request.messages)
        
        # 获取或创建会话
        session_id = await session_manager.get_session_for_conversation(conversation_id, xstech_model)
        
        # 检查是否需要切换模型
        current_model = session_manager.sessions[session_id]["model"]
        if current_model != xstech_model:
            logger.info(f"检测到模型切换请求: {current_model} -> {xstech_model}")
            model_updated = await xstech_client.update_session_model(session_id, xstech_model)
            if not model_updated:
                logger.warning(f"模型切换失败，将继续使用当前模型: {current_model}")
        
        # 获取会话锁，确保并发安全
        session_lock = await session_manager.get_session_lock(session_id)
        
        async with session_lock:
            # 最后一条消息内容
            last_message = request.messages[-1]
            user_text = ""
            files = []
            
            if isinstance(last_message.content, str):
                user_text = last_message.content
            elif isinstance(last_message.content, list):
                # 处理多模态内容
                for item in last_message.content:
                    if item.get("type") == "text":
                        user_text += item.get("text", "")
                
                # 处理图片
                files = process_image_content(last_message.content)
            
            # 获取会话关联的cookie_id，用于后续释放
            if session_id in session_manager.sessions:
                 cookie_id = session_manager.sessions[session_id]["cookie_id"]
            else:
                 # 这不应该发生，但作为保险措施
                 raise HTTPException(status_code=500, detail="Internal session inconsistency")

            # 发送请求到xstech，接收 session 和 response
            http_session, response = await xstech_client.send_chat_completion(session_id, user_text, files)
            
            # --- 开始处理响应 ---
            try:
                # 如果是流式输出
                if request.stream:
                    async def stream_generator():
                        ai_response = ""
                        usage_info = {}
                        stream_error = False
                        try:
                            # 处理SSE流
                            async for line in response.content:
                                line = line.decode('utf-8').strip()
                                logger.debug(f"Stream raw line: {line}") # Debug log
                                if not line:
                                    continue

                                if line.startswith("data: "):
                                    data_text = line[6:]
                                    if data_text == "[DONE]":
                                        logger.info(f"Stream finished for session {session_id}")
                                        break
                                    
                                    try:
                                        data = json.loads(data_text)
                                        logger.debug(f"Stream data: {data}") # Debug log
                                        
                                        # 处理字符串类型的数据
                                        if data.get("type") == "string" and "data" in data:
                                            chunk_content = data["data"]
                                            ai_response += chunk_content
                                            
                                            # 创建OpenAI格式的流式响应
                                            completion_chunk = {
                                                "id": f"chatcmpl-{str(uuid.uuid4())}",
                                                "object": "chat.completion.chunk",
                                                "created": int(time.time()),
                                                "model": request.model,
                                                "choices": [
                                                    {
                                                        "index": 0,
                                                        "delta": { "content": chunk_content },
                                                        "finish_reason": None
                                                    }
                                                ]
                                            }
                                            yield f"data: {json.dumps(completion_chunk)}\n\n"
                                        
                                        # 处理对象类型的数据（最后一条包含token使用情况）
                                        elif data.get("type") == "object" and "data" in data:
                                             # 记录最终的对象数据以备调试
                                            logger.info(f"Stream final object data: {data['data']}")
                                            usage_info = {
                                                "prompt_tokens": data["data"].get("promptTokens", 0),
                                                "completion_tokens": data["data"].get("completionTokens", 0),
                                                "total_tokens": data["data"].get("useTokens", 0)
                                            }
                                            
                                            # 发送最后一个chunk标记完成
                                            final_chunk = {
                                                "id": f"chatcmpl-{str(uuid.uuid4())}",
                                                "object": "chat.completion.chunk",
                                                "created": int(time.time()),
                                                "model": request.model,
                                                "choices": [
                                                    { "index": 0, "delta": {}, "finish_reason": "stop" }
                                                ]
                                            }
                                            yield f"data: {json.dumps(final_chunk)}\n\n"
                                            
                                    except json.JSONDecodeError:
                                        logger.error(f"Failed to parse SSE data: {data_text}")
                                    except Exception as parse_err:
                                        logger.error(f"Error processing SSE data chunk: {parse_err}", exc_info=True)
                                        stream_error = True
                                        break # Stop processing stream on error
                                else:
                                     logger.warning(f"Received non-SSE line: {line}") # Log unexpected lines

                            # 只有在流成功完成且有响应时才记录历史
                            if not stream_error and ai_response:
                                session_manager.sessions[session_id]["messages"].append({ "role": "user", "content": user_text })
                                session_manager.sessions[session_id]["messages"].append({ "role": "assistant", "content": ai_response })
                                
                        except aiohttp.ClientPayloadError as e:
                             logger.error(f"Stream payload error for session {session_id}: {e}", exc_info=True)
                             stream_error = True
                        except Exception as e:
                            logger.error(f"Error during stream processing for session {session_id}: {e}", exc_info=True)
                            stream_error = True
                        finally:
                             # 确保即使生成器提前退出也关闭连接和释放cookie
                            if http_session and not http_session.closed:
                                await http_session.close()
                                logger.info(f"HTTP session closed after stream for session {session_id}")
                            if cookie_id:
                                cookie_pool.release_cookie(cookie_id, error=stream_error)
                                logger.info(f"Cookie {cookie_id} released after stream for session {session_id}, error={stream_error}")
                            
                        yield "data: [DONE]\n\n" # 必须在finally之后发送 [DONE]
                        
                    return StreamingResponse(stream_generator(), media_type="text/event-stream")
                else:
                    # 处理非流式请求
                    ai_response = ""
                    usage_info = { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 }
                    non_stream_error = False
                    
                    try:
                        # 处理响应
                        async for line in response.content:
                            line = line.decode('utf-8').strip()
                            logger.debug(f"Non-stream raw line: {line}") # Debug log
                            if not line or not line.startswith("data: "):
                                continue
                            
                            data_text = line[6:]
                            if data_text == "[DONE]":
                                logger.info(f"Non-stream finished for session {session_id}")
                                break
                            
                            try:
                                data = json.loads(data_text)
                                logger.debug(f"Non-stream data: {data}") # Debug log
                                
                                # 处理字符串类型的数据
                                if data.get("type") == "string" and "data" in data:
                                    ai_response += data["data"]
                                
                                # 处理对象类型的数据（包含token使用情况）
                                elif data.get("type") == "object" and "data" in data:
                                    # 记录最终的对象数据以备调试
                                    logger.info(f"Non-stream final object data: {data['data']}")
                                    usage_info = {
                                        "prompt_tokens": data["data"].get("promptTokens", 0),
                                        "completion_tokens": data["data"].get("completionTokens", 0),
                                        "total_tokens": data["data"].get("useTokens", 0)
                                    }
                                    
                            except json.JSONDecodeError:
                                logger.error(f"Failed to parse non-stream response data: {data_text}")
                            except Exception as parse_err:
                                logger.error(f"Error processing non-stream data chunk: {parse_err}", exc_info=True)
                                non_stream_error = True
                                break # Stop processing on error
                    
                    except aiohttp.ClientPayloadError as e:
                         logger.error(f"Non-stream payload error for session {session_id}: {e}", exc_info=True)
                         non_stream_error = True
                    except Exception as e:
                        logger.error(f"Error during non-stream processing for session {session_id}: {e}", exc_info=True)
                        non_stream_error = True
                    finally:
                         # 确保关闭连接和释放cookie
                        if http_session and not http_session.closed:
                            await http_session.close()
                            logger.info(f"HTTP session closed after non-stream for session {session_id}")
                        if cookie_id:
                            cookie_pool.release_cookie(cookie_id, error=non_stream_error)
                            logger.info(f"Cookie {cookie_id} released after non-stream for session {session_id}, error={non_stream_error}")
                            
                    # 只有在处理成功且有响应时才记录历史
                    if not non_stream_error and ai_response:
                        session_manager.sessions[session_id]["messages"].append({ "role": "user", "content": user_text })
                        session_manager.sessions[session_id]["messages"].append({ "role": "assistant", "content": ai_response })
                    
                    # 返回OpenAI格式的响应
                    return {
                        "id": f"chatcmpl-{str(uuid.uuid4())}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "message": { "role": "assistant", "content": ai_response },
                                "finish_reason": "stop" if not non_stream_error else "error"
                            }
                        ],
                        "usage": usage_info
                    }
            
            except Exception as e:
                 # 处理响应过程中的其他异常
                logger.error(f"Error processing response for session {session_id}: {e}", exc_info=True)
                # 确保关闭和释放
                if http_session and not http_session.closed:
                    await http_session.close()
                if cookie_id:
                     cookie_pool.release_cookie(cookie_id, error=True)
                raise HTTPException(status_code=500, detail=f"Error processing response: {str(e)}")
                
    except HTTPException as e:
         # 对于已知HTTP异常，直接重新抛出
        raise e
    except Exception as e:
        # 捕获未预料的异常
        logger.error(f"Unhandled error in chat completion endpoint: {e}", exc_info=True)
        # 尝试释放cookie（如果已获取）
        if cookie_id:
             cookie_pool.release_cookie(cookie_id, error=True)
        # 尝试关闭http session（如果已创建）
        if http_session and not http_session.closed:
            await http_session.close()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# 健康检查端点
@app.get("/health")
async def health_check():
    """API健康检查"""
    return {"status": "healthy", "cookie_pool_size": len(cookie_pool.cookies)}

# 主函数
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("xstech_openai_api:app", host="0.0.0.0", port=8000, reload=True) 