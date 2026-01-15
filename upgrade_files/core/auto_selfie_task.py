from datetime import datetime
import time
import random
from typing import Optional, Tuple, Dict, Any, List

from src.manager.async_task_manager import AsyncTask
from src.common.logger import get_logger
from src.plugin_system.apis import send_api, generator_api
from src.plugin_system.apis.chat_api import get_chat_manager
from src.chat.message_receive.chat_stream import ChatStream
# 导入数据模型
from src.common.data_models.database_data_model import DatabaseMessages

logger = get_logger("auto_selfie_task")

class AutoSelfieTask(AsyncTask):
    """自动发送自拍定时任务"""

    def __init__(self, plugin_instance):
        # 从配置读取间隔时间
        config_interval_minutes = plugin_instance.get_config("auto_selfie.interval_minutes", 60)
        wait_seconds = config_interval_minutes * 60
        
        # 默认每分钟检查一次，具体是否发送由逻辑判断
        super().__init__(task_name="Auto Selfie Task", wait_before_start=wait_seconds, run_interval=60)
        
        self.plugin = plugin_instance
        self.last_send_time: Dict[str, float] = {}  # 记录每个群/用户的上次发送时间戳
        self.log_prefix = "[AutoSelfie]"  # 设置日志前缀

        # 检查全局任务中止标志（仅作检查，修复逻辑在plugin.py中）
        from src.manager.async_task_manager import async_task_manager
        if async_task_manager.abort_flag.is_set():
            logger.warning(f"[AutoSelfie] 全局任务中止标志 (abort_flag) 为 SET 状态，这可能会阻止任务运行。")

    async def run(self):
        """执行定时检查任务"""
        try:
            # 1. 检查总开关
            enabled = self.plugin.get_config("auto_selfie.enabled", False)
            if not enabled:
                return

            # 2. 检查当前是否在"麦麦睡觉"时间段
            if self._is_sleep_time():
                # 降低日志频率，仅在整点记录
                if datetime.now().minute == 0:
                    logger.debug("[AutoSelfie] 当前处于睡眠时间，跳过自拍")
                return

            # 3. 遍历所有活跃的聊天流
            # 使用新的 API 获取所有流
            from src.plugin_system.apis import chat_api
            streams = chat_api.get_all_streams(chat_api.SpecialTypes.ALL_PLATFORMS)
            
            # 如果 streams 为空，尝试从数据库加载 (这部分逻辑在 API 中可能已处理，但为了保险起见，如果 API 返回空列表，我们尝试手动加载)
            if not streams:
                try:
                    logger.info("[AutoSelfie] 内存中无活跃流，尝试从数据库加载所有流...")
                    chat_manager = get_chat_manager()
                    if hasattr(chat_manager, "load_all_streams"):
                        await chat_manager.load_all_streams()
                        # 重新获取
                        streams = chat_api.get_all_streams(chat_api.SpecialTypes.ALL_PLATFORMS)
                except Exception as e:
                    logger.error(f"[AutoSelfie] 加载流失败: {e}", exc_info=True)

            if not streams:
                return
            
            current_time = datetime.now().timestamp()
            interval_minutes = self.plugin.get_config("auto_selfie.interval_minutes", 60)
            interval_seconds = interval_minutes * 60

            # 获取白名单配置
            allowed_chat_ids = self.plugin.get_config("auto_selfie.allowed_chat_ids", [])
            # 确保是列表
            if not isinstance(allowed_chat_ids, list):
                allowed_chat_ids = []
            
            for stream in streams:
                stream_id = stream.stream_id
                
                # 如果配置了白名单，且当前流不在白名单中，则跳过
                if allowed_chat_ids:
                    # 构造可读的 ID 格式进行匹配
                    # 格式1: platform:user_id:private (私聊)
                    # 格式2: platform:group_id:group (群聊)
                    readable_ids = []
                    
                    try:
                        platform = getattr(stream, "platform", "unknown")
                        group_info = getattr(stream, "group_info", None)
                        if group_info:
                            # 群聊
                            group_id = str(getattr(group_info, "group_id", "unknown"))
                            readable_id = f"{platform}:{group_id}:group"
                            readable_ids.append(readable_id)
                            # 兼容旧格式：直接用 group_id
                            readable_ids.append(group_id)
                        elif getattr(stream, "user_info", None):
                            # 私聊
                            user_id = str(stream.user_info.user_id)
                            readable_id = f"{platform}:{user_id}:private"
                            readable_ids.append(readable_id)
                            # 兼容旧格式：直接用 user_id
                            readable_ids.append(user_id)
                    except Exception as e:
                        logger.warning(f"[AutoSelfie] 构建可读 ID 失败: {e}")

                    # 检查是否命中（支持哈希 ID 或可读 ID）
                    is_allowed = False
                    
                    # 1. 检查原始哈希 ID
                    if stream_id in allowed_chat_ids:
                        is_allowed = True
                    
                    # 2. 检查可读 ID
                    if not is_allowed:
                        for rid in readable_ids:
                            if rid in allowed_chat_ids:
                                is_allowed = True
                                logger.debug(f"[AutoSelfie] 流 {stream_id} 通过可读 ID {rid} 命中白名单")
                                break
                    
                    if not is_allowed:
                        logger.debug(f"[AutoSelfie] 流 {stream_id} (可读ID: {readable_ids}) 不在白名单中，跳过")
                        continue

                # 检查该流是否启用插件
                if not self._is_plugin_enabled_for_stream(stream_id):
                    continue

                # 检查时间间隔
                last_time = self.last_send_time.get(stream_id, 0)
                
                # 首次运行的处理逻辑：
                # 如果 last_time 为 0，说明插件刚刚启动或者这是新加载的流。
                # 我们不应该立即发送，而应该等待一个间隔，或者根据 create_time / last_active_time 判断。
                # 为了避免重启刷屏，我们设置首次运行时间为当前时间减去一个随机偏移量（让它在未来某个时间点触发）
                if last_time == 0:
                    # 初始化上次发送时间为当前时间，这样会等待一个周期后才开始发送
                    # 也可以选择随机等待 0 ~ interval 之间的时间
                    random_wait = random.uniform(0, interval_seconds)
                    
                    # 修正逻辑：我们希望在 random_wait (0~interval) 秒后触发
                    self.last_send_time[stream_id] = current_time + random_wait - interval_seconds
                    logger.info(f"[AutoSelfie] 流 {stream_id} 首次初始化，将在 {random_wait/60:.1f} 分钟后触发第一次自拍")
                    continue

                # 检查是否到达时间间隔
                if current_time - last_time >= interval_seconds:
                    # 增加一些随机性，避免所有群同时发（±20%的随机浮动）
                    random_offset = random.uniform(-0.2, 0.2) * interval_seconds
                    
                    # 只有当超过了 (间隔 + 随机偏移) 时才触发
                    if current_time - last_time >= interval_seconds + random_offset:
                        logger.info(f"[AutoSelfie] 流 {stream_id} 触发时间到，准备发送自拍")
                        await self._trigger_selfie_for_stream(stream) # 传入 stream 对象而非 ID，减少后续查找
                        self.last_send_time[stream_id] = current_time
                    else:
                        pass

        except Exception as e:
            logger.error(f"[AutoSelfie] 定时任务执行出错: {e}", exc_info=True)

    def _is_sleep_time(self) -> bool:
        """检查当前是否处于睡眠时间"""
        sleep_mode_enabled = self.plugin.get_config("auto_selfie.sleep_mode_enabled", True)
        if not sleep_mode_enabled:
            return False

        start_str = self.plugin.get_config("auto_selfie.sleep_start_time", "23:00")
        end_str = self.plugin.get_config("auto_selfie.sleep_end_time", "07:00")
        
        try:
            now_time = datetime.now().time()
            start_time = datetime.strptime(start_str, "%H:%M").time()
            end_time = datetime.strptime(end_str, "%H:%M").time()
            
            if start_time < end_time:
                # 例如 09:00 - 18:00 (白天睡觉？不常见但支持)
                return start_time <= now_time <= end_time
            else:
                # 跨夜，例如 23:00 - 07:00
                return now_time >= start_time or now_time <= end_time
        except Exception as e:
            logger.error(f"[AutoSelfie] 解析睡眠时间出错: {e}", exc_info=True)
            return False

    def _is_plugin_enabled_for_stream(self, stream_id: str) -> bool:
        """检查指定流是否启用插件"""
        # 暂时只检查全局配置
        try:
            from .runtime_state import runtime_state
            global_enabled = self.plugin.get_config("plugin.enabled", True)
            return runtime_state.is_plugin_enabled(stream_id, global_enabled)
        except ImportError:
            return self.plugin.get_config("plugin.enabled", True)

    async def _trigger_selfie_for_stream(self, stream_or_id):
        """为指定流触发自拍发送"""
        
        # 兼容旧版本传入 stream_id 的情况
        if isinstance(stream_or_id, str):
            stream_id = stream_or_id
            from src.plugin_system.apis import chat_api
            # 尝试通过 ID 获取 stream 对象
            chat_stream = None
            try:
                # 先尝试群聊
                chat_stream = chat_api.get_stream_by_group_id(stream_id.split(":")[1] if ":" in stream_id else stream_id)
            except:
                pass
            
            # 如果没找到，可能需要遍历（这里为了简化，假设传入的是 stream 对象）
            if not chat_stream:
                logger.warning(f"[AutoSelfie] 通过 ID {stream_id} 查找 stream 失败，尝试重新加载")
                # 重新获取所有流并查找
                streams = chat_api.get_all_streams()
                for s in streams:
                    if s.stream_id == stream_id:
                        chat_stream = s
                        break
        else:
            chat_stream = stream_or_id
            stream_id = chat_stream.stream_id

        if not chat_stream:
            logger.error(f"[AutoSelfie] 找不到流对象: {stream_id}")
            return

        logger.info(f"[AutoSelfie] 正在为 {stream_id} 触发定时自拍")
        
        try:
            # 1. 获取自拍配置
            style = self.plugin.get_config("auto_selfie.selfie_style", "standard")
            model_id = self.plugin.get_config("auto_selfie.model_id", "model1")
            use_replyer = self.plugin.get_config("auto_selfie.use_replyer_for_ask", True)
            
            # 2. 生成询问语
            ask_message = ""
            if use_replyer:
                # 使用 Replyer 生成自然的询问语
                prompt = "你刚刚拍了一张自拍发给对方。请生成一句简短、俏皮的询问语，问对方觉得好看吗，或者分享你此刻的心情。不要包含图片描述，只要询问语。50字以内。直接输出这句话，不要任何解释，不要说'好的'，不要给选项。"
                # 调用 generator_api 生成
                response = await generator_api.generate_response_custom(
                    chat_stream=chat_stream, # 优先使用 chat_stream
                    prompt=prompt,
                    # request_type="auto_selfie" # 移除非标准参数
                )
                ask_message = response if response else "你看这张照片怎么样？"
                # 清理可能残留的引号
                ask_message = ask_message.strip('"').strip("'")
            else:
                # 使用固定模板或配置
                config_ask = self.plugin.get_config("auto_selfie.ask_message", "")
                if config_ask:
                    ask_message = config_ask
                else:
                    templates = [
                        "你看这张照片怎么样？",
                        "刚刚随手拍的，好看吗？",
                        "分享一张此刻的我~",
                        "这是现在的我哦！",
                        "嘿嘿，来张自拍！"
                    ]
                    ask_message = random.choice(templates)

            # 3. 调用 Action 生成图片
            from .pic_action import Custom_Pic_Action
            
            # 构造虚拟消息对象 (DatabaseMessages) 用于 Action 初始化
            # 由于DatabaseMessages比较复杂且字段多变，我们尽可能提供必要的字段
            
            class MockUserInfo:
                def __init__(self, user_id, user_nickname, platform):
                    self.user_id = user_id
                    self.user_nickname = user_nickname
                    self.platform = platform
            
            class MockGroupInfo:
                def __init__(self, group_id, group_name, group_platform):
                    self.group_id = group_id
                    self.group_name = group_name
                    self.group_platform = group_platform

            class MockChatInfo:
                def __init__(self, platform, group_info=None):
                    self.platform = platform
                    self.group_info = group_info

            # 获取流信息
            # ChatStream 对象属性可能与数据库模型不同，需要做适配
            s_user_id = getattr(chat_stream.user_info, "user_id", "")
            s_user_nickname = getattr(chat_stream.user_info, "user_nickname", "User")
            s_platform = getattr(chat_stream, "platform", "unknown")
            
            is_group = False
            s_group_id = ""
            s_group_name = ""
            
            # 尝试判断是否群聊
            # 使用 getattr 安全获取属性
            if getattr(chat_stream, "is_group", False):
                is_group = True
                s_group_id = getattr(chat_stream, "group_id", "")
                s_group_name = getattr(chat_stream, "group_name", "")
            
            user_info = MockUserInfo(s_user_id, s_user_nickname, s_platform)
            group_info = MockGroupInfo(s_group_id, s_group_name, s_platform) if is_group else None
            chat_info = MockChatInfo(s_platform, group_info)
            
            # 构造 Mock Message
            mock_message = DatabaseMessages() # type: ignore
            mock_message.message_id = f"auto_selfie_{int(time.time())}"
            mock_message.time = time.time()
            mock_message.user_info = user_info # type: ignore
            mock_message.chat_info = chat_info # type: ignore
            mock_message.processed_plain_text = "auto selfie task"
            
            # 构造 action_data
            action_data = {
                "description": "auto selfie", 
                "model_id": model_id,
                "selfie_mode": True,
                "selfie_style": style,
                "size": "" 
            }
            
            # 实例化 Action
            action_instance = Custom_Pic_Action(
                action_data=action_data,
                action_reasoning="Auto selfie task triggered",
                cycle_timers={},
                thinking_id="auto_selfie",
                chat_stream=chat_stream,
                plugin_config=self.plugin.config, # 传入当前插件配置
                action_message=mock_message
            )
            
            # 4. 执行生成
            # (1) 生成提示词
            prompt = action_instance._process_selfie_prompt(
                description="a casual selfie", # 基础描述
                selfie_style=style,
                free_hand_action="",
                model_id=model_id
            )
            
            # (2) 获取负面提示词
            neg_prompt = self.plugin.get_config(f"selfie.negative_prompt_{style}", "")
            if not neg_prompt:
                neg_prompt = self.plugin.get_config("selfie.negative_prompt", "")
                
            # (3) 获取参考图（如果有）
            ref_image = action_instance._get_selfie_reference_image()
            
            # (4) 执行生成
            # 生成图片
            success, result = await action_instance._execute_unified_generation(
                description=prompt,
                model_id=model_id,
                size="",
                strength=0.6,
                input_image_base64=ref_image,
                extra_negative_prompt=neg_prompt
            )
            
            if success:
                logger.info(f"[AutoSelfie] 自拍发送成功: {stream_id}")
                
                # 发送询问语（在图片发送成功后）
                if ask_message:
                    # 稍微等待一下，让图片先展示
                    import asyncio
                    await asyncio.sleep(2)
                    await send_api.text_to_stream(ask_message, stream_id)
            else:
                logger.warning(f"[AutoSelfie] 自拍发送失败: {stream_id} - {result}")

        except Exception as e:
            logger.error(f"[AutoSelfie] 触发自拍失败: {e}", exc_info=True)
