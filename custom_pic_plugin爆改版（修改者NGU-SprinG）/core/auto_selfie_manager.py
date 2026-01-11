"""定时自拍任务管理器

按聊天流分别管理定时自拍任务，支持：
- 启动/停止定时自拍
- 自定义定时间隔
- 自定义自拍风格和模型
- 发完自拍后自动询问用户
"""
import asyncio
import random
from typing import Dict, Optional, Set
from dataclasses import dataclass, field
from src.common.logger import get_logger

logger = get_logger("auto_selfie_manager")


# 询问语模板
ASK_TEMPLATES = [
    "这张照片怎么样？",
    "你觉得这张自拍好看吗？",
    "今天的自拍还可以吗？",
    "这张照片拍得怎么样呀？",
    "你觉得我这张自拍怎么样？",
    "这张自拍拍得怎么样？",
    "今天的照片你喜欢吗？",
    "这张照片还可以吗？"
]


@dataclass
class AutoSelfieTask:
    """单个聊天流的定时自拍任务"""
    # 任务是否运行中
    is_running: bool = False
    # 定时间隔（分钟）
    interval_minutes: int = 30
    # 自拍风格
    selfie_style: str = "standard"
    # 使用的模型ID
    model_id: str = "model1"
    # 自定义询问语（空则使用随机模板）
    custom_ask_message: str = ""
    # 异步任务对象
    task: Optional[asyncio.Task] = None
    # 取消事件
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class PendingSelfieRequest:
    """待处理的自拍请求"""
    # 自拍风格
    selfie_style: str = "standard"
    # 使用的模型ID
    model_id: str = "model1"
    # 询问语
    ask_message: str = ""
    # 请求时间戳
    timestamp: float = 0.0


class AutoSelfieManager:
    """定时自拍任务管理器（单例）

    按聊天流ID分别管理定时自拍任务，所有状态仅在内存中保持，重启后重置。
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tasks: Dict[str, AutoSelfieTask] = {}
            cls._instance._pending_requests: Dict[str, PendingSelfieRequest] = {}
        return cls._instance

    def _get_task(self, chat_id: str) -> AutoSelfieTask:
        """获取或创建聊天流任务"""
        if chat_id not in self._tasks:
            self._tasks[chat_id] = AutoSelfieTask()
        return self._tasks[chat_id]

    def get_ask_message(self, custom_message: str = "") -> str:
        """获取询问语

        Args:
            custom_message: 自定义询问语，如果为空则随机选择模板

        Returns:
            询问语文本
        """
        if custom_message and custom_message.strip():
            return custom_message.strip()
        return random.choice(ASK_TEMPLATES)

    async def start_auto_selfie(
        self,
        chat_id: str,
        interval_minutes: int,
        selfie_style: str = "standard",
        model_id: str = "model1",
        custom_ask_message: str = "",
        action_callback=None
    ) -> bool:
        """启动定时自拍任务

        Args:
            chat_id: 聊天流ID
            interval_minutes: 定时间隔（分钟）
            selfie_style: 自拍风格
            model_id: 使用的模型ID
            custom_ask_message: 自定义询问语
            action_callback: 回调函数，用于触发自拍Action

        Returns:
            是否启动成功
        """
        task = self._get_task(chat_id)

        # 如果任务已在运行，先停止
        if task.is_running:
            logger.warning(f"[AutoSelfieManager] 聊天流 {chat_id} 定时自拍任务已在运行，先停止")
            await self.stop_auto_selfie(chat_id)

        # 更新任务配置
        task.interval_minutes = interval_minutes
        task.selfie_style = selfie_style
        task.model_id = model_id
        task.custom_ask_message = custom_ask_message
        task.is_running = True
        task.cancel_event.clear()

        logger.info(
            f"[AutoSelfieManager] 启动聊天流 {chat_id} 定时自拍任务: "
            f"间隔={interval_minutes}分钟, 风格={selfie_style}, 模型={model_id}"
        )

        # 创建并启动后台任务
        async def selfie_loop():
            """定时自拍循环"""
            try:
                while not task.cancel_event.is_set():
                    # 等待指定时间
                    try:
                        await asyncio.wait_for(
                            task.cancel_event.wait(),
                            timeout=task.interval_minutes * 60
                        )
                        # 如果被取消，退出循环
                        if task.cancel_event.is_set():
                            break
                    except asyncio.TimeoutError:
                        # 超时，执行自拍
                        pass

                    # 检查是否被取消
                    if task.cancel_event.is_set():
                        break

                    # 执行自拍
                    logger.info(f"[AutoSelfieManager] 聊天流 {chat_id} 执行定时自拍")
                    try:
                        if action_callback:
                            # 调用回调函数触发自拍
                            await action_callback(
                                chat_id=chat_id,
                                selfie_style=task.selfie_style,
                                model_id=task.model_id,
                                ask_message=self.get_ask_message(task.custom_ask_message)
                            )
                    except Exception as e:
                        logger.error(f"[AutoSelfieManager] 聊天流 {chat_id} 定时自拍执行失败: {e}", exc_info=True)

            except asyncio.CancelledError:
                logger.debug(f"[AutoSelfieManager] 聊天流 {chat_id} 定时自拍任务被取消")
            except Exception as e:
                logger.error(f"[AutoSelfieManager] 聊天流 {chat_id} 定时自拍任务异常: {e}", exc_info=True)
            finally:
                task.is_running = False
                logger.info(f"[AutoSelfieManager] 聊天流 {chat_id} 定时自拍任务已停止")

        task.task = asyncio.create_task(selfie_loop())
        return True

    async def stop_auto_selfie(self, chat_id: str) -> bool:
        """停止定时自拍任务

        Args:
            chat_id: 聊天流ID

        Returns:
            是否停止成功
        """
        task = self._get_task(chat_id)

        if not task.is_running:
            logger.info(f"[AutoSelfieManager] 聊天流 {chat_id} 定时自拍任务未运行")
            return False

        logger.info(f"[AutoSelfieManager] 停止聊天流 {chat_id} 定时自拍任务")

        # 设置取消事件
        task.cancel_event.set()

        # 取消任务
        if task.task and not task.task.done():
            task.task.cancel()
            try:
                await task.task
            except asyncio.CancelledError:
                pass

        task.is_running = False
        task.task = None
        return True

    def is_running(self, chat_id: str) -> bool:
        """检查定时自拍任务是否运行中

        Args:
            chat_id: 聊天流ID

        Returns:
            是否运行中
        """
        task = self._get_task(chat_id)
        return task.is_running

    def get_task_info(self, chat_id: str) -> Optional[Dict]:
        """获取定时自拍任务信息

        Args:
            chat_id: 聊天流ID

        Returns:
            任务信息字典，如果任务不存在则返回None
        """
        task = self._get_task(chat_id)
        if not task.is_running:
            return None

        return {
            "is_running": task.is_running,
            "interval_minutes": task.interval_minutes,
            "selfie_style": task.selfie_style,
            "model_id": task.model_id,
            "custom_ask_message": task.custom_ask_message
        }

    def get_all_running_tasks(self) -> Dict[str, Dict]:
        """获取所有运行中的定时自拍任务

        Returns:
            字典，键为chat_id，值为任务信息
        """
        result = {}
        for chat_id, task in self._tasks.items():
            if task.is_running:
                result[chat_id] = {
                    "interval_minutes": task.interval_minutes,
                    "selfie_style": task.selfie_style,
                    "model_id": task.model_id,
                    "custom_ask_message": task.custom_ask_message
                }
        return result

    def stop_all_tasks(self):
        """停止所有定时自拍任务"""
        logger.info(f"[AutoSelfieManager] 停止所有定时自拍任务")
        for chat_id in list(self._tasks.keys()):
            # 使用 asyncio.create_task 异步停止
            asyncio.create_task(self.stop_auto_selfie(chat_id))

    def add_pending_selfie_request(
        self,
        chat_id: str,
        selfie_style: str = "standard",
        model_id: str = "model1",
        ask_message: str = ""
    ):
        """添加待处理的自拍请求

        Args:
            chat_id: 聊天流ID
            selfie_style: 自拍风格
            model_id: 使用的模型ID
            ask_message: 询问语
        """
        import time
        self._pending_requests[chat_id] = PendingSelfieRequest(
            selfie_style=selfie_style,
            model_id=model_id,
            ask_message=ask_message,
            timestamp=time.time()
        )
        logger.info(f"[AutoSelfieManager] 添加待处理自拍请求: chat_id={chat_id}, style={selfie_style}, model={model_id}")

    def get_and_clear_pending_selfie_request(self, chat_id: str) -> Optional[PendingSelfieRequest]:
        """获取并清除待处理的自拍请求

        Args:
            chat_id: 聊天流ID

        Returns:
            待处理的自拍请求，如果不存在则返回None
        """
        request = self._pending_requests.pop(chat_id, None)
        if request:
            logger.info(f"[AutoSelfieManager] 获取待处理自拍请求: chat_id={chat_id}, style={request.selfie_style}, model={request.model_id}")
        return request

    def has_pending_selfie_request(self, chat_id: str) -> bool:
        """检查是否有待处理的自拍请求

        Args:
            chat_id: 聊天流ID

        Returns:
            是否有待处理的自拍请求
        """
        return chat_id in self._pending_requests


# 全局单例
auto_selfie_manager = AutoSelfieManager()
