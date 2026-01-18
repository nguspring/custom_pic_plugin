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

# 导入新的叙事模块
from .selfie_models import CaptionType, NarrativeScene, DailyNarrativeState
from .narrative_manager import NarrativeManager
from .caption_generator import CaptionGenerator

# 导入动态日程系统模块
from .schedule_models import DailySchedule, ScheduleEntry
from .schedule_generator import ScheduleGenerator
from .scene_action_generator import SceneActionGenerator

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
        
        # 默认每5分钟检查一次，具体是否发送由逻辑判断
        super().__init__(task_name="Auto Selfie Task", wait_before_start=wait_seconds, run_interval=300)
        
        self.plugin = plugin_instance
        self.last_send_time: Dict[str, float] = {}  # interval模式: 记录每个群/用户的上次发送时间戳
        self.last_send_dates: Dict[str, Dict[str, str]] = {} # times模式: 记录每个群/用户每个时间点的最后发送日期 {"stream_id": {"08:00": "2024-01-13"}}
        
        # DEBUG日志频率控制 (5分钟一次)
        self._last_debug_log_time: float = 0
        self._debug_log_interval: int = 300  # 5分钟 = 300秒

        # 加载状态
        self._load_state()
        
        # 初始化叙事管理器和配文生成器（用于 hybrid 模式和叙事配文功能）
        self.narrative_manager: Optional[NarrativeManager] = None
        self.caption_generator: Optional[CaptionGenerator] = None
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            narrative_state_path = os.path.join(os.path.dirname(current_dir), "narrative_state.json")
            logger.info(f"{self.log_prefix} [DEBUG-叙事] 正在初始化叙事管理器，状态文件路径: {narrative_state_path}")
            self.narrative_manager = NarrativeManager(plugin_instance, narrative_state_path)
            logger.info(f"{self.log_prefix} [DEBUG-叙事] 叙事管理器初始化成功，当前剧本: {self.narrative_manager.current_script_id}")
            self.caption_generator = CaptionGenerator(plugin_instance)
            logger.info(f"{self.log_prefix} [DEBUG-叙事] 配文生成器初始化成功")
            logger.info(f"{self.log_prefix} 叙事管理器和配文生成器初始化成功")
        except Exception as e:
            import traceback
            logger.warning(f"{self.log_prefix} 叙事模块初始化失败，将使用传统配文方式: {e}")
            logger.warning(f"{self.log_prefix} [DEBUG-叙事] 初始化失败堆栈: {traceback.format_exc()}")

        # 初始化动态日程系统（用于 smart 模式）
        self.schedule_generator: Optional[ScheduleGenerator] = None
        self.current_schedule: Optional[DailySchedule] = None
        self._schedule_lock = threading.Lock()
        try:
            self.schedule_generator = ScheduleGenerator(plugin_instance)
            logger.info(f"{self.log_prefix} 日程生成器初始化成功")
        except Exception as e:
            import traceback
            logger.warning(f"{self.log_prefix} 日程生成器初始化失败: {e}")
            logger.debug(f"{self.log_prefix} 初始化失败堆栈: {traceback.format_exc()}")

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
        """执行定时检查任务
        
        v3.6.0 更新：统一使用 Smart 模式（时间点 + LLM场景）
        - interval 模式已废弃，自动转换为 smart 模式
        - times 模式自动升级为 smart 模式
        - hybrid 模式自动升级为 smart 模式
        """
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

            # 3. 获取所有活跃的聊天流
            from src.plugin_system.apis import chat_api
            streams = chat_api.get_all_streams(chat_api.SpecialTypes.ALL_PLATFORMS)
            
            # 如果 streams 为空，尝试从数据库加载
            if not streams:
                try:
                    logger.info("[AutoSelfie] 内存中无活跃流，尝试从数据库加载所有流...")
                    chat_manager = get_chat_manager()
                    if hasattr(chat_manager, "load_all_streams"):
                        await chat_manager.load_all_streams()
                        streams = chat_api.get_all_streams(chat_api.SpecialTypes.ALL_PLATFORMS)
                except Exception as e:
                    logger.error(f"[AutoSelfie] 加载流失败: {e}", exc_info=True)

            if not streams:
                return
            
            current_time_obj = datetime.now()
            current_date_str = current_time_obj.strftime("%Y-%m-%d")
            
            # 获取调度模式配置
            schedule_mode = self.plugin.get_config("auto_selfie.schedule_mode", "smart")
            
            # ============================================================
            # v3.6.0 模式统一：所有模式都使用 Smart 模式处理
            # ============================================================
            
            # 处理模式迁移和废弃警告
            if schedule_mode == "interval":
                # Interval 模式已废弃，给出警告并自动转换
                logger.warning(
                    f"{self.log_prefix} [废弃警告] interval 模式已废弃！"
                    f"倒计时触发不符合'真人感'需求，建议修改配置为 schedule_mode = \"smart\"。"
                    f"系统将自动使用 smart 模式处理。"
                )
                schedule_mode = "smart"
            elif schedule_mode == "times":
                # Times 模式自动升级为 Smart 模式
                logger.info(
                    f"{self.log_prefix} [模式升级] times 模式已自动升级为 smart 模式，"
                    f"保留时间点触发，增强场景生成能力。"
                )
                schedule_mode = "smart"
            elif schedule_mode == "hybrid":
                # Hybrid 模式自动升级为 Smart 模式
                logger.info(
                    f"{self.log_prefix} [模式升级] hybrid 模式已自动升级为 smart 模式，"
                    f"移除倒计时补充，使用纯时间点+场景触发。"
                )
                schedule_mode = "smart"
            
            # 4. 筛选符合条件的白名单流
            allowed_streams = self._filter_allowed_streams(streams)
            
            if not allowed_streams:
                return
            
            # 5. 统一使用 Smart 模式处理
            logger.debug(f"{self.log_prefix} 符合条件的流数量: {len(allowed_streams)}，使用 Smart 模式")
            await self._process_smart_mode(allowed_streams, current_time_obj, current_date_str)

        except Exception as e:
            logger.error(f"[AutoSelfie] 定时任务执行出错: {e}", exc_info=True)

    def _filter_allowed_streams(self, streams: List) -> List:
        """筛选符合白名单/黑名单条件的流
        
        Args:
            streams: 所有聊天流列表
            
        Returns:
            符合条件的流列表
        """
        list_mode = self.plugin.get_config("auto_selfie.list_mode", "whitelist")
        chat_id_list = self.plugin.get_config("auto_selfie.chat_id_list", [])
        
        # 兼容旧配置
        if not chat_id_list:
            old_allowed = self.plugin.get_config("auto_selfie.allowed_chat_ids", [])
            if isinstance(old_allowed, list) and old_allowed:
                chat_id_list = old_allowed

        if not isinstance(chat_id_list, list):
            chat_id_list = []
        
        allowed_streams = []
        
        for stream in streams:
            stream_id = stream.stream_id
            is_allowed = False
            in_list = False
            
            # 如果列表为空，直接根据模式判断
            if not chat_id_list:
                if list_mode == "blacklist":
                    is_allowed = True
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
                    if in_list:
                        is_allowed = True
                else:  # blacklist
                    if not in_list:
                        is_allowed = True
            
            if not is_allowed:
                continue

            # 检查该流是否启用插件
            if not self._is_plugin_enabled_for_stream(stream_id):
                continue
            
            allowed_streams.append(stream)
        
        return allowed_streams

    async def _check_should_trigger(
        self,
        representative_stream,
        schedule_mode: str,
        target_times: List[str],
        current_time_obj: datetime,
        current_date_str: str,
        current_timestamp: float,
        interval_seconds: int
    ) -> Tuple[bool, str, Optional[str], Optional[str]]:
        """检查是否应该触发自拍
        
        Args:
            representative_stream: 代表流（用于状态检查）
            schedule_mode: 调度模式
            target_times: 目标时间点列表
            current_time_obj: 当前时间对象
            current_date_str: 当前日期字符串
            current_timestamp: 当前时间戳
            interval_seconds: 间隔秒数
            
        Returns:
            Tuple[是否触发, 触发类型, 场景描述, 匹配的时间点]
        """
        stream_id = representative_stream.stream_id
        current_hm = current_time_obj.strftime("%H:%M")
        
        # [DEBUG-叙事] 记录检查信息 (频率控制: 每5分钟一次)
        should_log_debug = (current_timestamp - self._last_debug_log_time) >= self._debug_log_interval
        if should_log_debug:
            self._last_debug_log_time = current_timestamp
            logger.info(f"{self.log_prefix} [DEBUG-叙事] === 触发检查 (每5分钟记录一次) ===")
            logger.info(f"{self.log_prefix} [DEBUG-叙事] 模式: {schedule_mode}, 当前时间: {current_hm}")
            logger.info(f"{self.log_prefix} [DEBUG-叙事] 目标时间点: {target_times}")
        
        # 解析自定义时间场景配置
        time_scenes = self._parse_time_scenes()
        
        # 处理 times 和 hybrid 模式的时间点触发
        if schedule_mode in ("times", "hybrid"):
            if stream_id not in self.last_send_dates:
                self.last_send_dates[stream_id] = {}
            
            for t_str in target_times:
                if ":" not in t_str or len(t_str) != 5:
                    continue
                
                # 检查是否已经发送过
                last_date = self.last_send_dates[stream_id].get(t_str, "")
                if last_date == current_date_str:
                    continue
                
                # 检查时间是否匹配 (±2分钟窗口)
                try:
                    t_hour, t_minute = map(int, t_str.split(':'))
                    target_dt = current_time_obj.replace(hour=t_hour, minute=t_minute, second=0, microsecond=0)
                    diff = abs((current_time_obj - target_dt).total_seconds())
                    
                    if diff < 120:
                        logger.info(f"{self.log_prefix} [Times] 触发时间点 {t_str} (误差 {diff:.1f}s)")
                        
                        # 获取场景描述
                        scene_description = None
                        if t_str in time_scenes:
                            scene_description = time_scenes[t_str]
                            logger.info(f"{self.log_prefix} 使用自定义时间场景: {t_str} -> {scene_description}")
                        else:
                            # 尝试从叙事管理器获取场景
                            if self.narrative_manager is not None:
                                try:
                                    current_scene = self.narrative_manager.get_current_scene()
                                    if current_scene:
                                        scene_description = current_scene.image_prompt
                                        logger.info(f"{self.log_prefix} 使用叙事场景: {current_scene.scene_id}")
                                except Exception as e:
                                    logger.warning(f"{self.log_prefix} 获取叙事场景失败: {e}")
                            
                            # 如果还没有场景，尝试 LLM 生成
                            if not scene_description:
                                enable_llm_scene = self.plugin.get_config("auto_selfie.enable_llm_scene", False)
                                if enable_llm_scene:
                                    scene_description = await self._generate_llm_scene()
                        
                        return True, "times", scene_description, t_str
                        
                except ValueError:
                    continue
        
        # 处理 interval 模式或 hybrid 模式的 interval 补充
        if schedule_mode == "interval" or (schedule_mode == "hybrid" and not self._is_near_times_point(current_hm, target_times, margin_minutes=30)):
            last_time = self.last_send_time.get(stream_id, 0)
            
            # 首次运行
            if last_time == 0:
                random_wait = random.uniform(0, interval_seconds)
                self.last_send_time[stream_id] = current_timestamp + random_wait - interval_seconds
                self._save_state()
                logger.info(f"{self.log_prefix} [Interval] 首次初始化，将在 {random_wait/60:.1f} 分钟后触发")
                return False, "", None, None
            
            # 检查是否到达间隔
            if current_timestamp - last_time >= interval_seconds:
                # hybrid 模式需要概率检查
                if schedule_mode == "hybrid":
                    probability = self.plugin.get_config("auto_selfie.interval_probability", 0.3)
                    if random.random() > probability:
                        logger.debug(f"{self.log_prefix} [Hybrid-Interval] 概率检查未通过 (p={probability})")
                        return False, "", None, None
                
                # 随机偏移
                random_offset = random.uniform(-0.2, 0.2) * interval_seconds
                if current_timestamp - last_time >= interval_seconds + random_offset:
                    logger.info(f"{self.log_prefix} [Interval] 触发间隔自拍")
                    
                    # 尝试 LLM 生成场景
                    scene_description = None
                    enable_llm_scene = self.plugin.get_config("auto_selfie.enable_llm_scene", False)
                    if enable_llm_scene:
                        scene_description = await self._generate_llm_scene()
                    
                    return True, "interval", scene_description, None
        
        return False, "", None, None

    async def _generate_selfie_content_once(
        self,
        representative_stream,
        description: Optional[str] = None,
        use_narrative_caption: bool = False
    ) -> Tuple[Optional[str], str, str]:
        """生成一次自拍图片和配文
        
        Args:
            representative_stream: 代表流（用于初始化 Action）
            description: 场景描述
            use_narrative_caption: 是否使用叙事配文
            
        Returns:
            Tuple[图片base64, 配文, 使用的prompt]
        """
        from .pic_action import CustomPicAction
        
        chat_stream = representative_stream
        stream_id = chat_stream.stream_id
        
        logger.info(f"{self.log_prefix} 开始生成自拍内容 (一次生成，多次发送)")
        
        try:
            # 1. 获取配置
            style = self.plugin.get_config("auto_selfie.selfie_style", "standard")
            model_id = self.plugin.get_config("auto_selfie.model_id", "model1")
            
            # 2. 生成配文
            ask_message = ""
            enable_narrative = self.plugin.get_config("auto_selfie.enable_narrative", True)
            
            if enable_narrative and use_narrative_caption and self.caption_generator is not None:
                try:
                    # 获取叙事上下文
                    current_scene: Optional[NarrativeScene] = None
                    narrative_context = ""
                    mood = "neutral"
                    
                    if self.narrative_manager is not None:
                        current_scene = self.narrative_manager.get_current_scene()
                        narrative_context = self.narrative_manager.get_narrative_context()
                        if self.narrative_manager.state is not None:
                            mood = self.narrative_manager.state.current_mood
                    
                    # 选择配文类型
                    caption_type = self.caption_generator.select_caption_type(
                        scene=current_scene,
                        narrative_context=narrative_context,
                        current_hour=datetime.now().hour
                    )
                    
                    # 确定场景描述
                    scene_desc = ""
                    if current_scene:
                        scene_desc = current_scene.description
                    elif description:
                        scene_desc = description
                    
                    # 生成配文
                    ask_message = await self.caption_generator.generate_caption(
                        caption_type=caption_type,
                        scene_description=scene_desc,
                        narrative_context=narrative_context,
                        image_prompt=description or "",
                        mood=mood
                    )
                    
                    # 标记场景完成
                    if current_scene and self.narrative_manager is not None and ask_message:
                        self.narrative_manager.mark_scene_completed(current_scene.scene_id, ask_message)
                    
                    logger.info(f"{self.log_prefix} 叙事配文生成成功: {ask_message}")
                    
                except Exception as e:
                    logger.warning(f"{self.log_prefix} 叙事配文生成失败: {e}")
                    ask_message = ""
            
            # 回退到传统方式
            if not ask_message:
                use_replyer = self.plugin.get_config("auto_selfie.use_replyer_for_ask", True)
                if use_replyer:
                    ask_message = await self._generate_traditional_caption(description)
                else:
                    templates = ["你看这张照片怎么样？", "刚刚随手拍的，好看吗？", "分享一张此刻的我~"]
                    ask_message = random.choice(templates)
            
            # 3. 生成图片
            # 构造 Mock 对象
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

            s_user_id = getattr(chat_stream.user_info, "user_id", "")
            s_user_nickname = getattr(chat_stream.user_info, "user_nickname", "User")
            s_platform = getattr(chat_stream, "platform", "unknown")
            
            is_group = getattr(chat_stream, "is_group", False)
            s_group_id = getattr(chat_stream, "group_id", "") if is_group else ""
            s_group_name = getattr(chat_stream, "group_name", "") if is_group else ""
            
            user_info = MockUserInfo(s_user_id, s_user_nickname, s_platform)
            group_info = MockGroupInfo(s_group_id, s_group_name, s_platform) if is_group else None
            chat_info = MockChatInfo(s_platform, group_info)
            
            mock_message = DatabaseMessages()  # type: ignore
            mock_message.message_id = f"auto_selfie_{int(time.time())}"
            mock_message.time = time.time()
            mock_message.user_info = user_info  # type: ignore[assignment]
            mock_message.chat_info = chat_info  # type: ignore[assignment]
            mock_message.processed_plain_text = "auto selfie task"
            
            action_data = {
                "description": "auto selfie",
                "model_id": model_id,
                "selfie_mode": True,
                "selfie_style": style,
                "size": ""
            }
            
            action_instance = CustomPicAction(
                action_data=action_data,
                action_reasoning="Auto selfie task triggered",
                cycle_timers={},
                thinking_id="auto_selfie",
                chat_stream=chat_stream,
                plugin_config=self.plugin.config,
                action_message=mock_message
            )
            
            # 生成提示词
            if description:
                base_description = description
            else:
                base_description = "a casual selfie"
            
            # 尝试优化提示词
            optimizer_enabled = self.plugin.get_config("prompt_optimizer.enabled", True)
            if optimizer_enabled and not description:
                try:
                    from .prompt_optimizer import optimize_prompt
                    success, optimized_prompt = await optimize_prompt(base_description, self.log_prefix)
                    if success:
                        base_description = optimized_prompt
                except Exception as e:
                    logger.warning(f"{self.log_prefix} 提示词优化失败: {e}")

            prompt = action_instance._process_selfie_prompt(
                description=base_description,
                selfie_style=style,
                free_hand_action="",
                model_id=model_id
            )
            
            # 获取负面提示词
            neg_prompt = self.plugin.get_config(f"selfie.negative_prompt_{style}", "")
            if not neg_prompt:
                neg_prompt = self.plugin.get_config("selfie.negative_prompt", "")
            
            # 获取参考图
            ref_image = action_instance._get_selfie_reference_image()
            
            # 执行图片生成（不发送，只获取 base64）
            image_base64 = await action_instance._generate_image_only(
                description=prompt,
                model_id=model_id,
                size="",
                strength=0.6,
                input_image_base64=ref_image,
                extra_negative_prompt=neg_prompt
            )
            
            return image_base64, ask_message, prompt
            
        except Exception as e:
            logger.error(f"{self.log_prefix} 生成自拍内容失败: {e}", exc_info=True)
            return None, "", ""

    async def _generate_traditional_caption(self, description: Optional[str]) -> str:
        """生成传统配文（非叙事模式）"""
        ask_model_id = self.plugin.get_config("auto_selfie.ask_model_id", "")
        
        if description:
            ask_prompt = f"""你刚刚拍了一张自拍，画面内容是：{description}。
请生成一句简短、俏皮的询问语，询问朋友们觉得这张照片怎么样。
要求：
1. 语气自然，符合年轻人社交风格
2. 可以提及照片中的场景或动作
3. 不超过30个字
4. 直接输出这句话，不要任何解释或前缀"""
        else:
            ask_prompt = "你刚刚拍了一张自拍发给对方。请生成一句简短、俏皮的询问语，问对方觉得好看吗。30字以内。直接输出这句话。"
        
        available_models = llm_api.get_available_models()
        
        # 选择模型
        model_config = None
        EXCLUDED_MODELS = {"embedding", "voice", "vlm", "lpmm_entity_extract", "lpmm_rdf_build"}
        PREFERRED_MODELS = ["replyer", "planner", "utils"]
        
        if ask_model_id and ask_model_id in available_models:
            model_config = available_models[ask_model_id]
        else:
            for model_name in PREFERRED_MODELS:
                if model_name in available_models:
                    model_config = available_models[model_name]
                    break
            if model_config is None:
                for model_name, config in available_models.items():
                    if model_name not in EXCLUDED_MODELS:
                        model_config = config
                        break
        
        if model_config:
            success, content, _, _ = await llm_api.generate_with_model(
                prompt=ask_prompt,
                model_config=model_config,
                request_type="plugin.auto_selfie.ask_generate",
                temperature=0.8,
                max_tokens=50
            )
            if success and content:
                return content.strip().strip('"').strip("'").strip()
        
        return "你看这张照片怎么样？"

    async def _send_image_to_stream(self, stream, image_base64: str):
        """发送图片到指定流
        
        Args:
            stream: 聊天流对象
            image_base64: 图片的 base64 编码
        """
        from src.plugin_system.apis import send_api
        
        stream_id = stream.stream_id
        
        # 使用 send_api.image_to_stream 发送图片（接收 base64 字符串）
        await send_api.image_to_stream(image_base64, stream_id)

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

    async def _process_hybrid_mode(
        self,
        stream,
        target_times: List[str],
        current_time_obj: datetime,
        current_date_str: str,
        current_timestamp: float,
        interval_seconds: int
    ):
        """处理混合模式
        
        混合模式的逻辑：
        1. 优先检查 times 模式的时间点（主线剧情）
        2. 如果不在任何 times 时间点附近，检查 interval 条件（补充内容）
        3. 使用共享的叙事状态
        4. interval 触发需要满足冷却条件，避免与 times 冲突
        """
        stream_id = stream.stream_id
        current_hm = current_time_obj.strftime("%H:%M")
        
        # [DEBUG-叙事] 记录 hybrid 模式入口信息 (频率控制: 每5分钟一次)
        current_ts = current_timestamp
        should_log_debug = (current_ts - self._last_debug_log_time) >= self._debug_log_interval
        
        if should_log_debug:
            self._last_debug_log_time = current_ts
            logger.info(f"{self.log_prefix} [DEBUG-叙事] === Hybrid模式检查开始 (每5分钟记录一次) ===")
            logger.info(f"{self.log_prefix} [DEBUG-叙事] 流ID: {stream_id}, 当前时间: {current_hm}, 日期: {current_date_str}")
            logger.info(f"{self.log_prefix} [DEBUG-叙事] 目标时间点: {target_times}")
            logger.info(f"{self.log_prefix} [DEBUG-叙事] NarrativeManager状态: {'已初始化' if self.narrative_manager is not None else '未初始化'}")
            logger.info(f"{self.log_prefix} [DEBUG-叙事] CaptionGenerator状态: {'已初始化' if self.caption_generator is not None else '未初始化'}")
            
            # 如果叙事管理器已初始化，输出更多状态信息
            if self.narrative_manager is not None:
                try:
                    state = self.narrative_manager.state
                    if state:
                        logger.info(f"{self.log_prefix} [DEBUG-叙事] 叙事状态 - 日期: {state.date}, 剧本: {state.script_id}")
                        logger.info(f"{self.log_prefix} [DEBUG-叙事] 叙事状态 - 已完成场景: {state.completed_scenes}")
                        logger.info(f"{self.log_prefix} [DEBUG-叙事] 叙事状态 - 当前情绪: {state.current_mood}")
                    else:
                        logger.warning(f"{self.log_prefix} [DEBUG-叙事] 叙事状态为 None")
                except Exception as e:
                    logger.warning(f"{self.log_prefix} [DEBUG-叙事] 读取叙事状态失败: {e}")
        
        # 步骤1: 检查是否有 times 模式的时间点需要触发
        times_triggered = await self._check_times_trigger(
            stream=stream,
            stream_id=stream_id,
            target_times=target_times,
            current_time_obj=current_time_obj,
            current_date_str=current_date_str
        )
        
        if times_triggered:
            return  # times 已触发，本轮结束
        
        # 步骤2: 检查 interval 补充触发条件
        interval_probability = self.plugin.get_config("auto_selfie.interval_probability", 0.3)
        
        # 只有当不在 times 时间点附近（±30分钟）时才考虑 interval
        if not self._is_near_times_point(current_hm, target_times, margin_minutes=30):
            await self._check_interval_supplement(
                stream=stream,
                stream_id=stream_id,
                current_timestamp=current_timestamp,
                interval_seconds=interval_seconds,
                probability=interval_probability
            )
        else:
            logger.debug(f"{self.log_prefix} 流 {stream_id} 在 times 时间点附近，跳过 interval 检查")

    def _is_near_times_point(self, current_hm: str, target_times: List[str], margin_minutes: int = 30) -> bool:
        """检查当前时间是否在任意 times 时间点附近
        
        Args:
            current_hm: 当前时间 "HH:MM"
            target_times: 时间点列表 ["HH:MM", ...]
            margin_minutes: 允许的分钟偏差
            
        Returns:
            bool: 如果在任意时间点附近返回 True
        """
        try:
            current_hour, current_minute = map(int, current_hm.split(":"))
            current_total_mins = current_hour * 60 + current_minute
            
            for t_str in target_times:
                if ":" not in t_str or len(t_str) != 5:
                    continue
                    
                try:
                    t_hour, t_minute = map(int, t_str.split(":"))
                    t_total_mins = t_hour * 60 + t_minute
                    
                    # 计算分钟差（考虑跨午夜情况）
                    diff = abs(current_total_mins - t_total_mins)
                    # 处理跨午夜情况 (例如 23:50 到 00:10 应该只差 20 分钟)
                    if diff > 720:  # 超过12小时，取另一个方向
                        diff = 1440 - diff
                    
                    if diff <= margin_minutes:
                        return True
                except ValueError:
                    continue
            
            return False
            
        except Exception as e:
            logger.warning(f"{self.log_prefix} 时间点检查失败: {e}")
            return False

    async def _check_times_trigger(
        self,
        stream,
        stream_id: str,
        target_times: List[str],
        current_time_obj: datetime,
        current_date_str: str
    ) -> bool:
        """检查并触发 times 模式，返回是否已触发
        
        Args:
            stream: 聊天流对象
            stream_id: 流 ID
            target_times: 时间点列表
            current_time_obj: 当前时间对象
            current_date_str: 当前日期字符串
            
        Returns:
            bool: 如果触发了发送返回 True
        """
        # 确保该流的时间记录存在
        if stream_id not in self.last_send_dates:
            self.last_send_dates[stream_id] = {}
        
        # 解析自定义时间场景配置
        time_scenes = self._parse_time_scenes()
        
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
                if diff < 120:
                    logger.info(f"{self.log_prefix} [Hybrid-Times] 流 {stream_id} 触发时间点 {t_str} (误差 {diff:.1f}s)")
                    
                    # 检查是否有自定义场景描述
                    scene_description: Optional[str] = None
                    if t_str in time_scenes:
                        scene_description = time_scenes[t_str]
                        logger.info(f"{self.log_prefix} 使用自定义时间场景: {t_str} -> {scene_description}")
                    else:
                        # 尝试使用叙事管理器获取场景
                        logger.info(f"{self.log_prefix} [DEBUG-叙事] 尝试从叙事管理器获取场景...")
                        if self.narrative_manager is not None:
                            try:
                                logger.info(f"{self.log_prefix} [DEBUG-叙事] 调用 narrative_manager.get_current_scene()")
                                current_scene = self.narrative_manager.get_current_scene()
                                if current_scene:
                                    scene_description = current_scene.image_prompt
                                    logger.info(f"{self.log_prefix} [DEBUG-叙事] 获取到叙事场景: scene_id={current_scene.scene_id}, description={current_scene.description}")
                                    logger.info(f"{self.log_prefix} [DEBUG-叙事] 场景时间范围: {current_scene.time_start} - {current_scene.time_end}")
                                    logger.info(f"{self.log_prefix} [DEBUG-叙事] 场景image_prompt: {current_scene.image_prompt}")
                                    logger.info(f"{self.log_prefix} 使用叙事场景: {current_scene.scene_id}")
                                else:
                                    logger.info(f"{self.log_prefix} [DEBUG-叙事] get_current_scene() 返回 None，当前时间不在任何场景范围内")
                            except Exception as e:
                                import traceback
                                logger.warning(f"{self.log_prefix} 获取叙事场景失败: {e}")
                                logger.warning(f"{self.log_prefix} [DEBUG-叙事] 获取场景失败堆栈: {traceback.format_exc()}")
                        else:
                            logger.warning(f"{self.log_prefix} [DEBUG-叙事] narrative_manager 为 None，无法获取叙事场景")
                        
                        # 如果还没有场景，检查是否启用 LLM 场景
                        if not scene_description:
                            enable_llm_scene = self.plugin.get_config("auto_selfie.enable_llm_scene", False)
                            if enable_llm_scene:
                                scene_description = await self._generate_llm_scene()
                    
                    # 发送（带场景描述，使用叙事配文）
                    await self._trigger_selfie_for_stream(
                        stream,
                        description=scene_description,
                        use_narrative_caption=True
                    )
                    
                    # 更新状态
                    self.last_send_dates[stream_id][t_str] = current_date_str
                    self._save_state()
                    return True
                    
            except Exception as e:
                logger.warning(f"{self.log_prefix} 时间点 {t_str} 处理失败: {e}")
                continue
        
        return False

    async def _check_interval_supplement(
        self,
        stream,
        stream_id: str,
        current_timestamp: float,
        interval_seconds: int,
        probability: float = 0.3
    ) -> bool:
        """检查并触发 interval 补充，返回是否已触发
        
        在 hybrid 模式下，interval 作为补充内容，有概率触发
        
        Args:
            stream: 聊天流对象
            stream_id: 流 ID
            current_timestamp: 当前时间戳
            interval_seconds: 间隔秒数
            probability: 触发概率 (0.0-1.0)
            
        Returns:
            bool: 如果触发了发送返回 True
        """
        # 检查时间间隔
        last_time = self.last_send_time.get(stream_id, 0)
        
        # 首次运行的处理逻辑
        if last_time == 0:
            random_wait = random.uniform(0, interval_seconds)
            self.last_send_time[stream_id] = current_timestamp + random_wait - interval_seconds
            self._save_state()
            logger.info(f"{self.log_prefix} [Hybrid-Interval] 流 {stream_id} 首次初始化")
            return False
        
        # 检查是否到达时间间隔
        if current_timestamp - last_time >= interval_seconds:
            # 概率检查
            if random.random() > probability:
                logger.debug(f"{self.log_prefix} [Hybrid-Interval] 流 {stream_id} 概率检查未通过 (p={probability})")
                return False
            
            # 增加一些随机性
            random_offset = random.uniform(-0.2, 0.2) * interval_seconds
            
            if current_timestamp - last_time >= interval_seconds + random_offset:
                logger.info(f"{self.log_prefix} [Hybrid-Interval] 流 {stream_id} 触发补充自拍")
                
                # 检查是否启用 LLM 智能场景判断
                scene_description: Optional[str] = None
                enable_llm_scene = self.plugin.get_config("auto_selfie.enable_llm_scene", False)
                if enable_llm_scene:
                    scene_description = await self._generate_llm_scene()
                
                await self._trigger_selfie_for_stream(
                    stream,
                    description=scene_description,
                    use_narrative_caption=True
                )
                self.last_send_time[stream_id] = current_timestamp
                self._save_state()
                return True
        
        return False

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
            # 优先级：用户配置 > replyer > planner > utils
            # 明确排除不适合文本生成的模型
            EXCLUDED_MODELS = {"embedding", "voice", "vlm", "lpmm_entity_extract", "lpmm_rdf_build"}
            PREFERRED_MODELS = ["replyer", "planner", "utils"]  # 按优先级排序
            
            model_config = None
            selected_model_name = ""
            
            if scene_llm_model and scene_llm_model in available_models:
                model_config = available_models[scene_llm_model]
                selected_model_name = scene_llm_model
                logger.debug(f"{self.log_prefix} 使用配置的 LLM 模型: {scene_llm_model}")
            else:
                # 按优先级尝试选择模型
                for model_name in PREFERRED_MODELS:
                    if model_name in available_models:
                        model_config = available_models[model_name]
                        selected_model_name = model_name
                        logger.debug(f"{self.log_prefix} 使用默认 LLM 模型: {model_name}")
                        break
                
                # 如果首选模型都不存在，从剩余模型中选择（排除不适合的）
                if model_config is None:
                    for model_name, config in available_models.items():
                        if model_name not in EXCLUDED_MODELS:
                            model_config = config
                            selected_model_name = model_name
                            logger.debug(f"{self.log_prefix} 使用备选 LLM 模型: {model_name}")
                            break
                
                # 如果仍然没有找到，记录警告
                if model_config is None:
                    logger.warning(f"{self.log_prefix} 无法找到适合文本生成的模型，可用模型: {list(available_models.keys())}")
            
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

    async def _trigger_selfie_for_stream(
        self,
        stream_or_id,
        *,
        description: Optional[str] = None,
        use_narrative_caption: bool = False
    ):
        """为指定流触发自拍发送
        
        Args:
            stream_or_id: ChatStream 对象或 stream_id 字符串
            description: 可选的场景描述，用于替代默认的 "a casual selfie"
            use_narrative_caption: 是否使用叙事配文系统（新功能）
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
            
            # 2. 生成询问语/配文
            ask_message = ""
            
            # 检查是否启用叙事配文系统
            enable_narrative = self.plugin.get_config("auto_selfie.enable_narrative", True)
            
            # [DEBUG-叙事] 记录叙事配文系统状态
            logger.info(f"{self.log_prefix} [DEBUG-叙事] === 配文生成检查 ===")
            logger.info(f"{self.log_prefix} [DEBUG-叙事] enable_narrative配置: {enable_narrative}")
            logger.info(f"{self.log_prefix} [DEBUG-叙事] use_narrative_caption参数: {use_narrative_caption}")
            logger.info(f"{self.log_prefix} [DEBUG-叙事] caption_generator状态: {'已初始化' if self.caption_generator is not None else '未初始化'}")
            logger.info(f"{self.log_prefix} [DEBUG-叙事] 条件判断: enable_narrative={enable_narrative} AND use_narrative_caption={use_narrative_caption} AND caption_generator存在={self.caption_generator is not None}")
            
            # 如果启用叙事配文且传入了 use_narrative_caption=True，优先使用新系统
            if enable_narrative and use_narrative_caption and self.caption_generator is not None:
                logger.info(f"{self.log_prefix} [DEBUG-叙事] ✓ 进入叙事配文系统")
                try:
                    # 使用新的配文生成系统
                    logger.debug(f"{self.log_prefix} 使用叙事配文系统生成配文")
                    
                    # 获取当前场景
                    current_scene: Optional[NarrativeScene] = None
                    narrative_context = ""
                    mood = "neutral"
                    
                    if self.narrative_manager is not None:
                        logger.info(f"{self.log_prefix} [DEBUG-叙事] 从叙事管理器获取场景和上下文...")
                        current_scene = self.narrative_manager.get_current_scene()
                        narrative_context = self.narrative_manager.get_narrative_context()
                        if self.narrative_manager.state is not None:
                            mood = self.narrative_manager.state.current_mood
                        logger.info(f"{self.log_prefix} [DEBUG-叙事] 当前场景: {current_scene.scene_id if current_scene else 'None'}")
                        logger.info(f"{self.log_prefix} [DEBUG-叙事] 叙事上下文长度: {len(narrative_context)} 字符")
                        logger.info(f"{self.log_prefix} [DEBUG-叙事] 当前情绪: {mood}")
                    else:
                        logger.warning(f"{self.log_prefix} [DEBUG-叙事] narrative_manager 为 None，无法获取叙事上下文")
                    
                    # 选择配文类型
                    logger.info(f"{self.log_prefix} [DEBUG-叙事] 调用 caption_generator.select_caption_type()...")
                    caption_type = self.caption_generator.select_caption_type(
                        scene=current_scene,
                        narrative_context=narrative_context,
                        current_hour=datetime.now().hour
                    )
                    logger.info(f"{self.log_prefix} [DEBUG-叙事] 选择的配文类型: {caption_type.value}")
                    
                    # 确定场景描述
                    scene_desc = ""
                    if current_scene:
                        scene_desc = current_scene.description
                    elif description:
                        scene_desc = description
                    logger.info(f"{self.log_prefix} [DEBUG-叙事] 场景描述: {scene_desc}")
                    
                    # 生成配文
                    logger.info(f"{self.log_prefix} [DEBUG-叙事] 调用 caption_generator.generate_caption()...")
                    ask_message = await self.caption_generator.generate_caption(
                        caption_type=caption_type,
                        scene_description=scene_desc,
                        narrative_context=narrative_context,
                        image_prompt=description or "",
                        mood=mood
                    )
                    logger.info(f"{self.log_prefix} [DEBUG-叙事] 生成的配文: {ask_message}")
                    
                    # 如果有场景，标记完成
                    if current_scene and self.narrative_manager is not None and ask_message:
                        logger.info(f"{self.log_prefix} [DEBUG-叙事] 标记场景 {current_scene.scene_id} 完成")
                        self.narrative_manager.mark_scene_completed(
                            current_scene.scene_id,
                            ask_message
                        )
                    
                    logger.info(f"{self.log_prefix} 叙事配文生成成功 (类型: {caption_type.value}): {ask_message}")
                    
                except Exception as e:
                    import traceback
                    logger.warning(f"{self.log_prefix} 叙事配文生成失败，回退到传统方式: {e}")
                    logger.warning(f"{self.log_prefix} [DEBUG-叙事] 配文生成失败堆栈: {traceback.format_exc()}")
                    ask_message = ""  # 重置，使用传统方式
            else:
                # [DEBUG-叙事] 记录为什么没有进入叙事配文系统
                reasons = []
                if not enable_narrative:
                    reasons.append("enable_narrative=False")
                if not use_narrative_caption:
                    reasons.append("use_narrative_caption=False")
                if self.caption_generator is None:
                    reasons.append("caption_generator未初始化")
                logger.info(f"{self.log_prefix} [DEBUG-叙事] ✗ 未进入叙事配文系统，原因: {', '.join(reasons)}")
            
            # 如果叙事配文失败或未启用，使用传统方式
            if not ask_message:
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

    # ============================================================
    # Smart 模式（动态日程系统）相关方法
    # ============================================================

    async def _ensure_daily_schedule(self) -> Optional[DailySchedule]:
        """确保当天日程已生成
        
        如果当前缓存的日程日期与今天不符，则重新生成或加载。
        
        Returns:
            当天的日程对象，失败返回 None
        """
        if self.schedule_generator is None:
            logger.warning(f"{self.log_prefix} 日程生成器未初始化，无法使用 smart 模式")
            return None
        
        today = datetime.now().strftime("%Y-%m-%d")
        
        # 检查缓存是否有效
        with self._schedule_lock:
            if self.current_schedule and self.current_schedule.date == today:
                logger.debug(f"{self.log_prefix} 使用缓存的日程: {today}")
                return self.current_schedule
        
        # 尝试加载或生成日程
        try:
            schedule_times = self.plugin.get_config(
                "auto_selfie.schedule_times", ["08:00", "12:00", "20:00"]
            )
            character_name = self.plugin.get_config(
                "auto_selfie.character_name", "麦麦"
            )
            character_persona = self.plugin.get_config(
                "auto_selfie.character_persona", "一个可爱的二次元女孩"
            )
            weather = self.plugin.get_config("auto_selfie.weather", "晴天")
            is_holiday = self.plugin.get_config("auto_selfie.is_holiday", False)
            
            logger.info(f"{self.log_prefix} [Smart] 开始加载或生成日程: {today}")
            
            schedule = await self.schedule_generator.get_or_generate_schedule(
                date=today,
                character_name=character_name,
                character_persona=character_persona,
                schedule_times=schedule_times,
                weather=weather,
                is_holiday=is_holiday,
            )
            
            if schedule:
                with self._schedule_lock:
                    self.current_schedule = schedule
                logger.info(
                    f"{self.log_prefix} [Smart] 日程加载成功，共 {len(schedule.entries)} 个条目"
                )
                return schedule
            else:
                logger.warning(f"{self.log_prefix} [Smart] 日程生成失败")
                return None
                
        except Exception as e:
            logger.error(f"{self.log_prefix} [Smart] 获取日程异常: {e}", exc_info=True)
            return None

    async def _process_smart_mode(
        self,
        allowed_streams: List,
        current_time_obj: datetime,
        current_date_str: str,
    ) -> None:
        """处理智能日程模式
        
        使用动态生成的日程来决定何时发送自拍以及发送什么内容。
        采用"生成一次，发送多次"模式，只调用一次 API 生成图片和配文，
        然后发送到所有白名单聊天流。
        
        v3.6.1 新增：支持随机间隔发送（作为日程的补充）
        
        Args:
            allowed_streams: 符合条件的聊天流列表
            current_time_obj: 当前时间对象
            current_date_str: 当前日期字符串
        """
        # [调试日志] 记录进入 Smart 模式和流数量
        # 这是"生成一次，发送多次"模式的入口点
        logger.info(
            f"{self.log_prefix} [Smart] ========== 开始处理智能日程模式 =========="
        )
        logger.info(
            f"{self.log_prefix} [Smart] 白名单流数量: {len(allowed_streams)}"
        )
        logger.info(
            f"{self.log_prefix} [Smart] 模式: '生成一次，发送多次' - 只调用一次API生成图片和配文"
        )
        
        # 初始化变量（确保在所有代码路径中都有定义）
        matched_time: Optional[str] = None
        fallback_key: str = f"smart_fallback_{current_date_str}"
        is_fallback_mode = False
        is_interval_trigger = False  # 是否是间隔补充触发
        current_timestamp = time.time()
        
        # 获取配置的时间点
        schedule_times = self.plugin.get_config(
            "auto_selfie.schedule_times", ["08:00", "12:00", "20:00"]
        )
        current_hm = current_time_obj.strftime("%H:%M")
        
        # 确保日程已生成
        schedule = await self._ensure_daily_schedule()
        current_entry: Optional[ScheduleEntry] = None
        
        if schedule:
            # 获取当前时间应该触发的条目
            current_entry = schedule.get_current_entry(current_time_obj)
            if current_entry:
                # 检查该条目是否已完成
                if current_entry.is_completed:
                    logger.debug(
                        f"{self.log_prefix} [Smart] 条目 {current_entry.time_point} 已完成"
                    )
                    # 条目已完成，但可能需要检查间隔补充
                    current_entry = None
                else:
                    logger.info(
                        f"{self.log_prefix} [Smart] 触发日程条目: {current_entry.time_point} "
                        f"({current_entry.activity_description})"
                    )
            else:
                # 当前时间没有匹配的日程条目，但可能需要检查间隔补充
                logger.debug(f"{self.log_prefix} [Smart] 当前时间没有匹配的日程条目")
        else:
            # 日程系统失败，尝试使用回退机制
            is_fallback_mode = True
            logger.warning(
                f"{self.log_prefix} [Smart] 日程系统不可用，尝试使用回退机制 "
                f"(仍采用'生成一次，发送多次'模式)"
            )
            
            # 检查是否在任何时间点的 ±2分钟窗口内
            for t_str in schedule_times:
                if ":" not in t_str or len(t_str) != 5:
                    continue
                try:
                    t_hour, t_minute = map(int, t_str.split(':'))
                    target_dt = current_time_obj.replace(hour=t_hour, minute=t_minute, second=0, microsecond=0)
                    diff = abs((current_time_obj - target_dt).total_seconds())
                    if diff < 120:  # 2分钟窗口
                        matched_time = t_str
                        break
                except ValueError:
                    continue
            
            if matched_time:
                # 检查是否今天已经在这个时间点发送过
                if fallback_key not in self.last_send_dates:
                    self.last_send_dates[fallback_key] = {}
                
                if self.last_send_dates[fallback_key].get(matched_time) == current_date_str:
                    logger.debug(
                        f"{self.log_prefix} [Smart-Fallback] 时间点 {matched_time} 今天已发送"
                    )
                    matched_time = None  # 重置，继续检查间隔补充
                else:
                    logger.info(
                        f"{self.log_prefix} [Smart-Fallback] 触发回退模式发送，时间点: {matched_time}"
                    )
        
        # ================================================================
        # 检查间隔补充触发（日程触发之外的随机发送）
        # ================================================================
        if not current_entry and not matched_time:
            # 日程和回退都没有触发，检查间隔补充
            enable_interval = self.plugin.get_config("auto_selfie.enable_interval_supplement", True)
            
            if enable_interval:
                # 检查是否在任何时间点附近（±30分钟），如果是则不触发间隔
                is_near_time_point = self._is_near_times_point(current_hm, schedule_times, margin_minutes=30)
                
                if not is_near_time_point:
                    # 获取间隔配置
                    interval_minutes = self.plugin.get_config("auto_selfie.interval_minutes", 120)
                    interval_probability = self.plugin.get_config("auto_selfie.interval_probability", 0.3)
                    interval_seconds = interval_minutes * 60
                    
                    # 使用全局间隔状态键（不是每个流独立的）
                    interval_key = "smart_interval_global"
                    last_interval_time = self.last_send_time.get(interval_key, 0)
                    
                    # 首次运行初始化
                    if last_interval_time == 0:
                        random_wait = random.uniform(0, interval_seconds)
                        self.last_send_time[interval_key] = current_timestamp + random_wait - interval_seconds
                        self._save_state()
                        logger.info(
                            f"{self.log_prefix} [Smart-Interval] 首次初始化，"
                            f"将在 {random_wait/60:.1f} 分钟后开始检查间隔触发"
                        )
                    elif current_timestamp - last_interval_time >= interval_seconds:
                        # 到达间隔时间，进行概率检查
                        if random.random() <= interval_probability:
                            # 增加一些随机偏移（±20%）
                            random_offset = random.uniform(-0.2, 0.2) * interval_seconds
                            if current_timestamp - last_interval_time >= interval_seconds + random_offset:
                                is_interval_trigger = True
                                logger.info(
                                    f"{self.log_prefix} [Smart-Interval] 触发间隔补充发送 "
                                    f"(间隔: {interval_minutes}分钟, 概率: {interval_probability*100:.0f}%)"
                                )
                        else:
                            logger.debug(
                                f"{self.log_prefix} [Smart-Interval] 概率检查未通过 "
                                f"(p={interval_probability*100:.0f}%)"
                            )
                else:
                    logger.debug(
                        f"{self.log_prefix} [Smart-Interval] 在时间点附近（±30分钟），跳过间隔检查"
                    )
            
            # 如果都没有触发，直接返回
            if not is_interval_trigger:
                logger.debug(f"{self.log_prefix} [Smart] 无触发条件，等待下一次检查")
                return
        
        # ================================================================
        # 关键：使用第一个流作为代表生成图片和配文（只生成一次）
        # ================================================================
        representative_stream = allowed_streams[0]
        
        # 确定触发类型
        trigger_type = "日程条目" if current_entry else ("回退时间点" if matched_time else "间隔补充")
        logger.info(
            f"{self.log_prefix} [Smart] 【生成阶段】触发类型: {trigger_type}"
        )
        logger.info(
            f"{self.log_prefix} [Smart] 【生成阶段】使用代表流 {representative_stream.stream_id} 生成内容"
        )
        logger.info(
            f"{self.log_prefix} [Smart] 【重要】只调用一次图片生成API，然后复用结果发送到所有流"
        )
        
        # 生成图片和配文（只调用一次）
        if current_entry:
            # 使用日程条目生成
            logger.info(f"{self.log_prefix} [Smart] 使用日程条目模式生成内容")
            image_base64, caption, prompt_used = await self._generate_selfie_content_with_entry(
                representative_stream=representative_stream,
                schedule_entry=current_entry,
            )
        elif is_interval_trigger:
            # 间隔补充模式：使用 LLM 生成场景
            logger.info(f"{self.log_prefix} [Smart-Interval] 使用间隔补充模式生成内容")
            scene_description = await self._generate_llm_scene()
            image_base64, caption, prompt_used = await self._generate_selfie_content_once(
                representative_stream=representative_stream,
                description=scene_description,
                use_narrative_caption=True,
            )
        else:
            # 回退模式：使用回退方式生成（无日程条目）
            logger.info(f"{self.log_prefix} [Smart] 使用回退模式生成内容")
            image_base64, caption, prompt_used = await self._generate_selfie_content_once(
                representative_stream=representative_stream,
                description=None,
                use_narrative_caption=True,
            )
        
        if not image_base64:
            logger.warning(f"{self.log_prefix} [Smart] 图片生成失败，取消发送")
            return
        
        # ================================================================
        # 关键：发送到所有符合条件的流（使用相同的图片和配文）
        # ================================================================
        logger.info(
            f"{self.log_prefix} [Smart] 【发送阶段】图片生成成功，开始发送到 {len(allowed_streams)} 个流"
        )
        logger.info(
            f"{self.log_prefix} [Smart] 【重要】以下所有流将收到相同的图片和配文"
        )
        
        # 记录所有目标流
        stream_ids = [s.stream_id for s in allowed_streams]
        logger.info(f"{self.log_prefix} [Smart] 目标流列表: {stream_ids}")
        
        # 发送到所有符合条件的流（使用相同的图片和配文）
        success_count = 0
        for idx, stream in enumerate(allowed_streams, 1):
            stream_id = stream.stream_id
            try:
                logger.info(f"{self.log_prefix} [Smart] [{idx}/{len(allowed_streams)}] 发送到流: {stream_id}")
                
                # 发送图片
                await self._send_image_to_stream(stream, image_base64)
                
                # 发送配文
                if caption:
                    import asyncio
                    await asyncio.sleep(2)
                    await send_api.text_to_stream(caption, stream_id)
                
                success_count += 1
                logger.info(f"{self.log_prefix} [Smart] [{idx}/{len(allowed_streams)}] 发送成功: {stream_id}")
                
            except Exception as e:
                logger.error(f"{self.log_prefix} [Smart] [{idx}/{len(allowed_streams)}] 发送失败: {stream_id} - {e}")
        
        logger.info(
            f"{self.log_prefix} [Smart] ========== 自拍发送完成 =========="
        )
        logger.info(
            f"{self.log_prefix} [Smart] 发送结果: 成功 {success_count}/{len(allowed_streams)} 个流"
        )
        
        # 标记完成并保存
        if current_entry and schedule:
            # 正常模式：标记日程条目完成
            schedule.mark_entry_completed(current_entry.time_point, caption)
            if self.schedule_generator is not None:
                schedule.save_to_file(self.schedule_generator.get_schedule_file_path())
            
            # 更新缓存
            with self._schedule_lock:
                self.current_schedule = schedule
            logger.debug(f"{self.log_prefix} [Smart] 日程条目已标记完成: {current_entry.time_point}")
        elif is_fallback_mode and matched_time:
            # 回退模式：记录发送状态
            if fallback_key not in self.last_send_dates:
                self.last_send_dates[fallback_key] = {}
            self.last_send_dates[fallback_key][matched_time] = current_date_str
            self._save_state()
            logger.debug(f"{self.log_prefix} [Smart-Fallback] 发送状态已保存: {matched_time}")
        elif is_interval_trigger:
            # 间隔补充模式：更新间隔计时器
            interval_key = "smart_interval_global"
            self.last_send_time[interval_key] = current_timestamp
            self._save_state()
            logger.debug(f"{self.log_prefix} [Smart-Interval] 间隔计时器已更新")

    async def _generate_selfie_content_with_entry(
        self,
        representative_stream,
        schedule_entry: ScheduleEntry,
    ) -> Tuple[Optional[str], str, str]:
        """使用日程条目生成自拍图片和配文
        
        Args:
            representative_stream: 代表流（用于初始化 Action）
            schedule_entry: 日程条目
            
        Returns:
            Tuple[图片base64, 配文, 使用的prompt]
        """
        from .pic_action import CustomPicAction
        
        chat_stream = representative_stream
        stream_id = chat_stream.stream_id
        
        logger.info(f"{self.log_prefix} [Smart] 使用日程条目生成自拍内容")
        
        try:
            # 1. 获取配置
            style = self.plugin.get_config("auto_selfie.selfie_style", "standard")
            model_id = self.plugin.get_config("auto_selfie.model_id", "model1")
            
            # 2. 使用 SceneActionGenerator 生成配文上下文
            scene_generator = SceneActionGenerator(self.plugin)
            caption_context = scene_generator.create_caption_context(schedule_entry)
            
            # 3. 生成配文
            caption = await self._generate_caption_for_entry(schedule_entry, caption_context)
            
            # 4. 构造 Mock 对象
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

            s_user_id = getattr(chat_stream.user_info, "user_id", "")
            s_user_nickname = getattr(chat_stream.user_info, "user_nickname", "User")
            s_platform = getattr(chat_stream, "platform", "unknown")
            
            is_group = getattr(chat_stream, "is_group", False)
            s_group_id = getattr(chat_stream, "group_id", "") if is_group else ""
            s_group_name = getattr(chat_stream, "group_name", "") if is_group else ""
            
            user_info = MockUserInfo(s_user_id, s_user_nickname, s_platform)
            group_info = MockGroupInfo(s_group_id, s_group_name, s_platform) if is_group else None
            chat_info = MockChatInfo(s_platform, group_info)
            
            mock_message = DatabaseMessages()  # type: ignore
            mock_message.message_id = f"smart_selfie_{int(time.time())}"
            mock_message.time = time.time()
            mock_message.user_info = user_info  # type: ignore[assignment]
            mock_message.chat_info = chat_info  # type: ignore[assignment]
            mock_message.processed_plain_text = "smart selfie task"
            
            action_data = {
                "description": "smart selfie",
                "model_id": model_id,
                "selfie_mode": True,
                "selfie_style": style,
                "size": ""
            }
            
            action_instance = CustomPicAction(
                action_data=action_data,
                action_reasoning="Smart schedule selfie triggered",
                cycle_timers={},
                thinking_id="smart_selfie",
                chat_stream=chat_stream,
                plugin_config=self.plugin.config,
                action_message=mock_message
            )
            
            # 5. 使用场景驱动方式生成提示词
            prompt = action_instance._process_selfie_prompt(
                description="",  # 空描述，完全由 schedule_entry 驱动
                selfie_style=style,
                free_hand_action="",
                model_id=model_id,
                schedule_entry=schedule_entry,  # 传入日程条目
            )
            
            # 6. 获取负面提示词
            neg_prompt = scene_generator.get_negative_prompt_for_style(style)
            
            # 7. 获取参考图
            ref_image = action_instance._get_selfie_reference_image()
            
            # 8. 执行图片生成
            image_base64 = await action_instance._generate_image_only(
                description=prompt,
                model_id=model_id,
                size="",
                strength=0.6,
                input_image_base64=ref_image,
                extra_negative_prompt=neg_prompt
            )
            
            return image_base64, caption, prompt
            
        except Exception as e:
            logger.error(
                f"{self.log_prefix} [Smart] 生成自拍内容失败: {e}", exc_info=True
            )
            return None, "", ""

    async def _generate_caption_for_entry(
        self,
        schedule_entry: ScheduleEntry,
        caption_context: Dict[str, str],
    ) -> str:
        """为日程条目生成配文
        
        Args:
            schedule_entry: 日程条目
            caption_context: 配文上下文
            
        Returns:
            生成的配文
        """
        # 首先尝试使用叙事配文系统
        if self.caption_generator is not None:
            try:
                # 将 caption_type 字符串转换为 CaptionType 枚举
                caption_type_str = schedule_entry.caption_type.upper()
                try:
                    caption_type = CaptionType(caption_type_str.lower())
                except ValueError:
                    caption_type = CaptionType.SHARE
                
                caption = await self.caption_generator.generate_caption(
                    caption_type=caption_type,
                    scene_description=schedule_entry.activity_description,
                    narrative_context=caption_context.get("activity_detail", ""),
                    image_prompt=schedule_entry.suggested_caption_theme,
                    mood=schedule_entry.mood,
                )
                
                if caption:
                    logger.info(
                        f"{self.log_prefix} [Smart] 配文生成成功 (类型: {caption_type.value})"
                    )
                    return caption
                    
            except Exception as e:
                logger.warning(f"{self.log_prefix} [Smart] 叙事配文生成失败: {e}")
        
        # 回退到传统方式
        return await self._generate_traditional_caption(schedule_entry.activity_description)
