from datetime import datetime
import time
import random
import json
import os
import threading
from typing import Optional, Tuple, Dict, Any, List

from src.manager.async_task_manager import AsyncTask
from src.common.logger import get_logger
from src.plugin_system.apis import send_api, generator_api, llm_api
from src.plugin_system.apis.chat_api import get_chat_manager
from src.chat.message_receive.chat_stream import ChatStream
# 导入数据模型
from src.common.data_models.database_data_model import DatabaseMessages
# 导入 bot 名称配置
from src.config.config import global_config

logger = get_logger("auto_selfie_task")

class AutoSelfieTask(AsyncTask):
    """自动发送自拍定时任务"""

    def __init__(self, plugin_instance):
        self.log_prefix = "[AutoSelfie]"  # 设置日志前缀
        
        # 初始化持久化文件路径
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 存放在插件根目录
            self.state_file_path = os.path.join(os.path.dirname(current_dir), "auto_selfie_state.json")
        except Exception as e:
            logger.error(f"{self.log_prefix} 初始化持久化路径失败: {e}", exc_info=True)
            self.state_file_path = ""

        self.file_lock = threading.Lock()

        # 检查是否存在状态文件
        has_state = False
        if self.state_file_path and os.path.exists(self.state_file_path):
            has_state = True

        # 从配置读取间隔时间
        config_interval_minutes = plugin_instance.get_config("auto_selfie.interval_minutes", 60)
        
        if has_state:
            # 如果存在状态文件，说明是重启，快速启动以恢复计时
            wait_seconds = 10
            logger.info(f"{self.log_prefix} 检测到持久化状态文件，将在 10 秒后启动检查")
        else:
            # 首次运行，保持原有的等待逻辑
            wait_seconds = config_interval_minutes * 60
        
        # 默认每分钟检查一次，具体是否发送由逻辑判断
        super().__init__(task_name="Auto Selfie Task", wait_before_start=wait_seconds, run_interval=60)
        
        self.plugin = plugin_instance
        self.last_send_time: Dict[str, float] = {}  # interval模式: 记录每个群/用户的上次发送时间戳
        self.last_send_dates: Dict[str, Dict[str, str]] = {} # times模式: 记录每个群/用户每个时间点的最后发送日期 {"stream_id": {"08:00": "2024-01-13"}}

        # 加载状态
        self._load_state()

        # 检查全局任务中止标志（仅作检查，修复逻辑在plugin.py中）
        from src.manager.async_task_manager import async_task_manager
        if async_task_manager.abort_flag.is_set():
            logger.warning(f"[AutoSelfie] 全局任务中止标志 (abort_flag) 为 SET 状态，这可能会阻止任务运行。")

    def _load_state(self):
        """从文件加载状态"""
        if not self.state_file_path or not os.path.exists(self.state_file_path):
            return

        try:
            with open(self.state_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    # 检查数据版本
                    if "version" in data and data["version"] >= 2:
                        self.last_send_time = data.get("interval", {})
                        self.last_send_dates = data.get("times", {})
                        logger.info(f"{self.log_prefix} 已加载持久化状态 (v2)，Interval记录: {len(self.last_send_time)}条, Times记录: {len(self.last_send_dates)}条")
                    else:
                        # 兼容旧版本格式 (直接是 interval 字典)
                        self.last_send_time = data
                        self.last_send_dates = {}
                        logger.info(f"{self.log_prefix} 已加载旧版持久化状态，共 {len(self.last_send_time)} 条记录")
        except json.JSONDecodeError:
            logger.warning(f"{self.log_prefix} 状态文件损坏，将使用空状态")
        except Exception as e:
            logger.error(f"{self.log_prefix} 加载状态失败: {e}", exc_info=True)

    def _save_state(self):
        """保存状态到文件"""
        if not self.state_file_path:
            return
            
        try:
            with self.file_lock:
                save_data = {
                    "version": 2,
                    "interval": self.last_send_time,
                    "times": self.last_send_dates
                }
                with open(self.state_file_path, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"{self.log_prefix} 保存状态失败: {e}", exc_info=True)

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
            
            current_time_obj = datetime.now()
            current_timestamp = current_time_obj.timestamp()
            current_date_str = current_time_obj.strftime("%Y-%m-%d")
            
            # 获取调度模式配置
            schedule_mode = self.plugin.get_config("auto_selfie.schedule_mode", "interval")
            target_times = []
            
            if schedule_mode == "times":
                raw_times = self.plugin.get_config("auto_selfie.schedule_times", ["08:00", "12:00", "20:00"])
                if isinstance(raw_times, list) and raw_times:
                    target_times = raw_times
                else:
                    logger.warning(f"{self.log_prefix} schedule_times 配置无效，回退到 interval 模式")
                    schedule_mode = "interval"

            interval_minutes = self.plugin.get_config("auto_selfie.interval_minutes", 60)
            interval_seconds = interval_minutes * 60

            # 获取名单配置
            list_mode = self.plugin.get_config("auto_selfie.list_mode", "whitelist")
            chat_id_list = self.plugin.get_config("auto_selfie.chat_id_list", [])
            
            # 兼容旧配置：如果新列表为空但旧白名单有值，则读取旧白名单
            if not chat_id_list:
                old_allowed = self.plugin.get_config("auto_selfie.allowed_chat_ids", [])
                if isinstance(old_allowed, list) and old_allowed:
                    chat_id_list = old_allowed

            # 确保是列表
            if not isinstance(chat_id_list, list):
                chat_id_list = []
            
            for stream in streams:
                stream_id = stream.stream_id
                
                is_allowed = False
                in_list = False
                
                # 优化：如果列表为空，直接根据模式判断
                if not chat_id_list:
                    if list_mode == "blacklist":
                        is_allowed = True
                    # whitelist 默认为 False (留空则默认不允许)
                else:
                    # 列表不为空，需检查匹配
                    readable_ids = self._get_readable_ids(stream)
                    if stream_id in chat_id_list:
                        in_list = True
                    else:
                        for rid in readable_ids:
                            if rid in chat_id_list:
                                in_list = True
                                break
                    
                    if list_mode == "whitelist":
                        # 白名单模式：在列表中才允许
                        if in_list:
                            is_allowed = True
                            logger.debug(f"[AutoSelfie] 流 {stream_id} 命中白名单，允许发送")
                    else:
                        # 黑名单模式：不在列表中才允许
                        if not in_list:
                            is_allowed = True
                        else:
                            logger.debug(f"[AutoSelfie] 流 {stream_id} 命中黑名单，禁止发送")
                
                if not is_allowed:
                    continue

                # 检查该流是否启用插件
                if not self._is_plugin_enabled_for_stream(stream_id):
                    continue

                # 根据模式执行调度
                if schedule_mode == "times":
                    await self._process_times_mode(stream, target_times, current_time_obj, current_date_str)
                else:
                    await self._process_interval_mode(stream, stream_id, current_timestamp, interval_seconds)

        except Exception as e:
            logger.error(f"[AutoSelfie] 定时任务执行出错: {e}", exc_info=True)

    def _get_readable_ids(self, stream) -> List[str]:
        """获取流的可读 ID 列表"""
        readable_ids = []
        try:
            platform = getattr(stream, "platform", "unknown")
            group_info = getattr(stream, "group_info", None)
            if group_info:
                # 群聊
                group_id = str(getattr(group_info, "group_id", "unknown"))
                readable_ids.append(f"{platform}:{group_id}:group")
                # 兼容旧格式
                readable_ids.append(group_id)
            elif getattr(stream, "user_info", None):
                # 私聊
                user_id = str(stream.user_info.user_id)
                readable_ids.append(f"{platform}:{user_id}:private")
                # 兼容旧格式
                readable_ids.append(user_id)
        except Exception as e:
            logger.warning(f"[AutoSelfie] 构建可读 ID 失败: {e}")
        return readable_ids

    def _parse_time_scenes(self) -> Dict[str, str]:
        """解析 time_scenes 配置为字典
        
        配置格式: ["HH:MM|场景描述", ...]
        返回格式: {"HH:MM": "场景描述", ...}
        
        Returns:
            Dict[str, str]: 时间点到场景描述的映射
        """
        time_scenes_config = self.plugin.get_config("auto_selfie.time_scenes", [])
        result: Dict[str, str] = {}
        
        if not isinstance(time_scenes_config, list):
            logger.warning(f"{self.log_prefix} time_scenes 配置格式无效，应为列表类型")
            return result
        
        for item in time_scenes_config:
            if not isinstance(item, str):
                logger.warning(f"{self.log_prefix} time_scenes 配置项格式无效，应为字符串: {item}")
                continue
            
            if "|" not in item:
                logger.warning(f"{self.log_prefix} time_scenes 配置项缺少分隔符 '|': {item}")
                continue
            
            try:
                time_str, scene = item.split("|", 1)
                time_str = time_str.strip()
                scene = scene.strip()
                
                # 验证时间格式 (HH:MM)
                if ":" not in time_str or len(time_str) != 5:
                    logger.warning(f"{self.log_prefix} time_scenes 时间格式无效，应为 HH:MM: {time_str}")
                    continue
                
                # 验证时间合法性
                hour, minute = map(int, time_str.split(":"))
                if not (0 <= hour <= 23 and 0 <= minute <= 59):
                    logger.warning(f"{self.log_prefix} time_scenes 时间值超出范围: {time_str}")
                    continue
                
                if scene:
                    result[time_str] = scene
                    logger.debug(f"{self.log_prefix} 已解析时间场景: {time_str} -> {scene}")
                else:
                    logger.warning(f"{self.log_prefix} time_scenes 场景描述为空: {item}")
                    
            except ValueError as e:
                logger.warning(f"{self.log_prefix} time_scenes 配置项解析失败: {item}, 错误: {e}")
                continue
        
        if result:
            logger.info(f"{self.log_prefix} 已加载 {len(result)} 个自定义时间场景")
        
        return result

    async def _process_times_mode(self, stream, target_times: List[str], current_time_obj: datetime, current_date_str: str):
        """处理指定时间点模式"""
        stream_id = stream.stream_id
        
        # 确保该流的时间记录存在
        if stream_id not in self.last_send_dates:
            self.last_send_dates[stream_id] = {}
        
        # 解析自定义时间场景配置
        time_scenes = self._parse_time_scenes()
            
        current_hm = current_time_obj.strftime("%H:%M")
        
        for t_str in target_times:
            # 1. 简单验证格式
            if ":" not in t_str or len(t_str) != 5:
                continue
                
            # 2. 检查是否已经发送过
            last_date = self.last_send_dates[stream_id].get(t_str, "")
            if last_date == current_date_str:
                continue
                
            # 3. 检查时间是否匹配 (考虑前后 2 分钟窗口)
            try:
                # 解析目标时间
                t_hour, t_minute = map(int, t_str.split(':'))
                # 构造当天的目标时间
                target_dt = current_time_obj.replace(hour=t_hour, minute=t_minute, second=0, microsecond=0)
                
                # 计算时间差（秒）
                diff = abs((current_time_obj - target_dt).total_seconds())
                
                # 允许 120 秒 (2分钟) 的误差窗口
                # 这样即使任务调度有延迟，或者刚才错过了几十秒，也能补发
                if diff < 120:
                    logger.info(f"[AutoSelfie] 流 {stream_id} 触发时间点 {t_str} (误差 {diff:.1f}s)，准备发送自拍")
                    
                    # 检查是否有自定义场景描述
                    scene_description: Optional[str] = None
                    if t_str in time_scenes:
                        scene_description = time_scenes[t_str]
                        logger.info(f"{self.log_prefix} 使用自定义时间场景: {t_str} -> {scene_description}")
                    else:
                        # 没有自定义场景时，检查是否启用 LLM 智能场景判断
                        enable_llm_scene = self.plugin.get_config("auto_selfie.enable_llm_scene", False)
                        if enable_llm_scene:
                            scene_description = await self._generate_llm_scene()
                            if scene_description:
                                logger.info(f"{self.log_prefix} Times模式: LLM 生成场景描述: {scene_description}")
                    
                    # 发送（带场景描述）
                    await self._trigger_selfie_for_stream(stream, description=scene_description)
                    
                    # 更新状态
                    self.last_send_dates[stream_id][t_str] = current_date_str
                    self._save_state()
                    # 一次 run 只触发一个时间点即可
                    break
            except Exception as e:
                # 解析时间出错忽略
                continue

    async def _process_interval_mode(self, stream, stream_id: str, current_timestamp: float, interval_seconds: int):
        """处理倒计时模式"""
        # 检查时间间隔
        last_time = self.last_send_time.get(stream_id, 0)
        
        # 首次运行的处理逻辑
        if last_time == 0:
            random_wait = random.uniform(0, interval_seconds)
            self.last_send_time[stream_id] = current_timestamp + random_wait - interval_seconds
            self._save_state()
            logger.info(f"[AutoSelfie] 流 {stream_id} 首次初始化，将在 {random_wait/60:.1f} 分钟后触发第一次自拍")
            return

        # 检查是否到达时间间隔
        if current_timestamp - last_time >= interval_seconds:
            # 增加一些随机性，避免所有群同时发（±20%的随机浮动）
            random_offset = random.uniform(-0.2, 0.2) * interval_seconds
            
            if current_timestamp - last_time >= interval_seconds + random_offset:
                logger.info(f"[AutoSelfie] 流 {stream_id} 触发时间到 (Interval)，准备发送自拍")
                
                # [新增] 检查是否启用 LLM 智能场景判断
                scene_description: Optional[str] = None
                enable_llm_scene = self.plugin.get_config("auto_selfie.enable_llm_scene", False)
                if enable_llm_scene:
                    scene_description = await self._generate_llm_scene()
                    if scene_description:
                        logger.info(f"{self.log_prefix} LLM 生成场景描述: {scene_description}")
                
                await self._trigger_selfie_for_stream(stream, description=scene_description)
                self.last_send_time[stream_id] = current_timestamp
                self._save_state()

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

    async def _generate_llm_scene(self) -> Optional[str]:
        """使用 LLM 根据当前时间生成自拍场景描述
        
        Returns:
            Optional[str]: 生成的场景描述，失败时返回 None
        """
        try:
            # 获取当前时间信息
            now = datetime.now()
            time_str = now.strftime("%H:%M")
            date_str = now.strftime("%Y-%m-%d")
            
            # 获取星期几（中英文）
            weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            weekday = weekday_names[now.weekday()]
            
            # 获取 bot 名称（使用 nickname 属性）
            try:
                bot_name = global_config.bot.nickname
                if not bot_name:
                    bot_name = "MaiBot"
            except Exception:
                bot_name = "MaiBot"
            
            # 构建提示词
            prompt = f"""Current time: {time_str} on {date_str} ({weekday}).
Character: {bot_name}.
Task: Describe a selfie scene for {bot_name} at this moment.
Requirements:
1. Describe the location, outfit, and action suitable for the current time.
2. Keep it concise, suitable for Stable Diffusion prompts (English).
3. Format: "location, outfit, action".
4. No explanations, just the prompt tags.

Example output for morning: "cozy bedroom, pajamas, stretching, morning sunlight"
Example output for afternoon: "cafe, casual dress, holding coffee cup, relaxed smile"
Example output for evening: "home office, casual wear, looking at camera, soft lamp light"

Now generate for current time ({time_str}):"""

            # 获取模型配置
            scene_llm_model = self.plugin.get_config("auto_selfie.scene_llm_model", "")
            
            # 获取可用模型
            available_models = llm_api.get_available_models()
            
            if not available_models:
                logger.warning(f"{self.log_prefix} 无可用的 LLM 模型，无法生成场景")
                return None
            
            # 选择模型配置
            model_config = None
            if scene_llm_model and scene_llm_model in available_models:
                model_config = available_models[scene_llm_model]
                logger.debug(f"{self.log_prefix} 使用配置的 LLM 模型: {scene_llm_model}")
            else:
                # 使用默认模型（优先选择 normal_chat 或第一个可用模型）
                if "normal_chat" in available_models:
                    model_config = available_models["normal_chat"]
                    logger.debug(f"{self.log_prefix} 使用默认 LLM 模型: normal_chat")
                else:
                    # 使用第一个可用模型
                    first_model_name = next(iter(available_models))
                    model_config = available_models[first_model_name]
                    logger.debug(f"{self.log_prefix} 使用第一个可用 LLM 模型: {first_model_name}")
            
            if not model_config:
                logger.warning(f"{self.log_prefix} 未找到可用的 LLM 模型配置")
                return None
            
            # 调用 LLM 生成场景描述
            success, content, reasoning, model_name = await llm_api.generate_with_model(
                prompt=prompt,
                model_config=model_config,
                request_type="plugin.auto_selfie.scene_generate",
                temperature=0.8,  # 稍高的温度以增加创意性
                max_tokens=100    # 场景描述不需要太长
            )
            
            if success and content:
                # 清理返回结果（去除引号、换行等）
                scene = content.strip().strip('"').strip("'").strip()
                # 移除可能的前缀解释
                if ":" in scene and len(scene.split(":")[0]) < 20:
                    scene = scene.split(":", 1)[-1].strip()
                logger.info(f"{self.log_prefix} LLM 场景生成成功 (使用模型: {model_name}): {scene}")
                return scene
            else:
                logger.warning(f"{self.log_prefix} LLM 场景生成失败: {content}")
                return None
                
        except Exception as e:
            logger.error(f"{self.log_prefix} LLM 场景生成出错: {e}", exc_info=True)
            return None

    async def _trigger_selfie_for_stream(self, stream_or_id, *, description: Optional[str] = None):
        """为指定流触发自拍发送
        
        Args:
            stream_or_id: ChatStream 对象或 stream_id 字符串
            description: 可选的场景描述，用于替代默认的 "a casual selfie"
        """
        
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
                # 获取 ask_model_id 配置
                ask_model_id = self.plugin.get_config("auto_selfie.ask_model_id", "")
                
                # 构建优化后的 Prompt，包含场景描述
                if description:
                    ask_prompt = f"""你刚刚拍了一张自拍，画面内容是：{description}。
请生成一句简短、俏皮的询问语，询问朋友们觉得这张照片怎么样。
要求：
1. 语气自然，符合年轻人社交风格
2. 可以提及照片中的场景或动作
3. 不超过30个字
4. 直接输出这句话，不要任何解释或前缀"""
                    logger.debug(f"{self.log_prefix} 询问语 Prompt 包含场景描述: {description}")
                else:
                    ask_prompt = "你刚刚拍了一张自拍发给对方。请生成一句简短、俏皮的询问语，问对方觉得好看吗，或者分享你此刻的心情。不要包含图片描述，只要询问语。30字以内。直接输出这句话，不要任何解释，不要说'好的'，不要给选项。"
                
                # 获取可用模型列表
                available_models = llm_api.get_available_models()
                
                # 查找指定模型配置
                model_config = None
                if ask_model_id and available_models:
                    if ask_model_id in available_models:
                        model_config = available_models[ask_model_id]
                        logger.info(f"{self.log_prefix} 询问语生成使用配置的模型: {ask_model_id}")
                    else:
                        logger.warning(f"{self.log_prefix} 配置的询问语模型 '{ask_model_id}' 不存在，使用默认模型")
                
                # 如果没有指定模型或未找到，使用默认模型
                if model_config is None and available_models:
                    if "normal_chat" in available_models:
                        model_config = available_models["normal_chat"]
                        logger.debug(f"{self.log_prefix} 询问语生成使用默认模型: normal_chat")
                    else:
                        first_model_name = next(iter(available_models))
                        model_config = available_models[first_model_name]
                        logger.debug(f"{self.log_prefix} 询问语生成使用第一个可用模型: {first_model_name}")
                
                if model_config:
                    # 使用 llm_api 生成询问语
                    success, content, reasoning, model_name = await llm_api.generate_with_model(
                        prompt=ask_prompt,
                        model_config=model_config,
                        request_type="plugin.auto_selfie.ask_generate",
                        temperature=0.8,
                        max_tokens=50
                    )
                    
                    if success and content:
                        ask_message = content.strip().strip('"').strip("'").strip()
                        logger.info(f"{self.log_prefix} 询问语生成成功 (模型: {model_name}): {ask_message}")
                    else:
                        logger.warning(f"{self.log_prefix} 询问语生成失败，使用默认询问语")
                        ask_message = "你看这张照片怎么样？"
                else:
                    logger.warning(f"{self.log_prefix} 无可用的 LLM 模型，使用默认询问语")
                    ask_message = "你看这张照片怎么样？"
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
            from .pic_action import CustomPicAction
            
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
            action_instance = CustomPicAction(
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
            # 如果传入了 description（来自 LLM 场景生成），则使用它；否则使用默认值
            if description:
                base_description = description
                logger.info(f"{self.log_prefix} 使用 LLM 生成的场景描述: {base_description}")
            else:
                base_description = "a casual selfie"
            
            # 尝试优化提示词（仅在没有 LLM 场景时优化，避免重复处理）
            optimizer_enabled = self.plugin.get_config("prompt_optimizer.enabled", True)
            if optimizer_enabled and not description:
                try:
                    from .prompt_optimizer import optimize_prompt
                    success, optimized_prompt = await optimize_prompt(base_description, self.log_prefix)
                    if success:
                        base_description = optimized_prompt
                        logger.info(f"{self.log_prefix} 定时自拍提示词优化成功: {base_description}")
                except Exception as e:
                    logger.warning(f"{self.log_prefix} 定时自拍提示词优化失败: {e}")

            prompt = action_instance._process_selfie_prompt(
                description=base_description, # 使用优化后的描述
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
