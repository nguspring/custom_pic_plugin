from datetime import datetime
import time
import random
import json
import os
import threading
from typing import Optional, Tuple, Dict, List

from src.manager.async_task_manager import AsyncTask
from src.common.logger import get_logger
from src.plugin_system.apis import send_api, llm_api
from src.plugin_system.apis.chat_api import get_chat_manager

# 导入数据模型
# 导入 bot 名称配置
from src.config.config import global_config

# 导入配文模块
from .selfie_models import CaptionType
from .caption_generator import CaptionGenerator

# 导入动态日程系统模块
from .schedule_models import DailySchedule, ScheduleEntry, SceneVariation
from .schedule_generator import ScheduleGenerator
from .scene_action_generator import SceneActionGenerator

# 导入共享工具
from .shared_utils import create_mock_message

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
        self.last_send_dates: Dict[
            str, Dict[str, str]
        ] = {}  # times模式: 记录每个群/用户每个时间点的最后发送日期 {"stream_id": {"08:00": "2024-01-13"}}

        # DEBUG日志频率控制 (5分钟一次)
        self._last_debug_log_time: float = 0
        self._debug_log_interval: int = 300  # 5分钟 = 300秒

        # 加载状态
        self._load_state()

        # 初始化配文生成器（用于 Smart 模式）
        self.caption_generator: Optional[CaptionGenerator] = None
        try:
            self.caption_generator = CaptionGenerator(plugin_instance)
            logger.info(f"{self.log_prefix} 配文生成器初始化成功")
        except Exception as e:
            import traceback

            logger.warning(f"{self.log_prefix} 配文生成器初始化失败，将使用传统配文方式: {e}")
            logger.debug(f"{self.log_prefix} 初始化失败堆栈: {traceback.format_exc()}")

        # 初始化动态日程系统（用于 Smart 模式）
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
            logger.warning("[AutoSelfie] 全局任务中止标志 (abort_flag) 为 SET 状态，这可能会阻止任务运行。")

    def _load_state(self):
        """从文件加载状态"""
        if not self.state_file_path or not os.path.exists(self.state_file_path):
            return

        try:
            with open(self.state_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    # 检查数据版本
                    if "version" in data and data["version"] >= 2:
                        self.last_send_time = data.get("interval", {})
                        self.last_send_dates = data.get("times", {})
                        logger.info(
                            f"{self.log_prefix} 已加载持久化状态 (v2)，Interval记录: {len(self.last_send_time)}条, Times记录: {len(self.last_send_dates)}条"
                        )
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
                save_data = {"version": 2, "interval": self.last_send_time, "times": self.last_send_dates}
                with open(self.state_file_path, "w", encoding="utf-8") as f:
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

    async def _generate_selfie_content_once(
        self, representative_stream, description: Optional[str] = None, use_narrative_caption: bool = False
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

        logger.info(f"{self.log_prefix} 开始生成自拍内容 (一次生成，多次发送)")

        try:
            # 1. 获取配置
            style = self.plugin.get_config("auto_selfie.selfie_style", "standard")
            model_id = self.plugin.get_config("auto_selfie.model_id", "model1")

            # 2. 配文生成准备（Phase 4：配文贴图需要等图片生成后拿到视觉摘要）
            ask_message = ""
            enable_narrative = self.plugin.get_config("auto_selfie.enable_narrative", True)

            selected_caption_type: Optional[CaptionType] = None
            if enable_narrative and use_narrative_caption and self.caption_generator is not None:
                try:
                    selected_caption_type = self.caption_generator.select_caption_type(
                        scene=None,  # Smart 模式不使用旧版 NarrativeScene
                        narrative_context="",
                        current_hour=datetime.now().hour,
                    )
                except Exception as e:
                    logger.warning(f"{self.log_prefix} 配文类型选择失败: {e}")
                    selected_caption_type = None

            # 3. 生成图片
            # 使用共享工具创建 Mock 对象
            mock_message = create_mock_message(chat_stream, "auto_selfie")

            action_data = {
                "description": "auto selfie",
                "model_id": model_id,
                "selfie_mode": True,
                "selfie_style": style,
                "size": "",
            }

            action_instance = CustomPicAction(
                action_data=action_data,
                action_reasoning="Auto selfie task triggered",
                cycle_timers={},
                thinking_id="auto_selfie",
                chat_stream=chat_stream,
                plugin_config=self.plugin.config,
                action_message=mock_message,
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
                description=base_description, selfie_style=style, free_hand_action="", model_id=model_id
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
                extra_negative_prompt=neg_prompt,
            )

            if not image_base64:
                return None, "", ""

            # Phase 4：VLM 视觉摘要 -> 配文贴图
            visual_summary = await self._generate_visual_summary_for_image(
                image_base64=image_base64,
                scene_hint=str(description or base_description or ""),
            )

            # 可选：一致性自检（仅当启用时）
            scene_desc_for_caption = str(description or base_description or "")
            if visual_summary:
                try:
                    is_consistent = await self._is_visual_summary_consistent(
                        planned_scene=scene_desc_for_caption,
                        visual_summary=visual_summary,
                    )
                    if not is_consistent:
                        logger.warning(
                            f"{self.log_prefix} [VLM] 视觉摘要与计划场景不一致，将以视觉摘要作为配文场景描述"
                        )
                        scene_desc_for_caption = visual_summary
                except Exception as e:
                    logger.warning(f"{self.log_prefix} 一致性自检失败，跳过: {e}")

            # 生成配文（基于视觉摘要；失败则回退）
            if (
                enable_narrative
                and use_narrative_caption
                and self.caption_generator is not None
                and selected_caption_type is not None
            ):
                try:
                    ask_message = await self.caption_generator.generate_caption(
                        caption_type=selected_caption_type,
                        scene_description=scene_desc_for_caption,
                        narrative_context="",
                        image_prompt=base_description,
                        mood="neutral",
                        visual_summary=visual_summary,
                    )
                except Exception as e:
                    logger.warning(f"{self.log_prefix} 配文生成失败: {e}")
                    ask_message = ""

            if not ask_message:
                use_replyer = self.plugin.get_config("auto_selfie.use_replyer_for_ask", True)
                if use_replyer:
                    ask_message = await self._generate_traditional_caption(visual_summary or description)
                else:
                    templates = ["你看这张照片怎么样？", "刚刚随手拍的，好看吗？", "分享一张此刻的我~"]
                    ask_message = random.choice(templates)

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
            ask_prompt = (
                "你刚刚拍了一张自拍发给对方。请生成一句简短、俏皮的询问语，问对方觉得好看吗。30字以内。直接输出这句话。"
            )

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
                max_tokens=50,
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

    def _infer_image_format_from_base64(self, image_base64: str) -> str:
        """根据 base64 前缀推断图片格式（用于 VLM）。"""
        if not image_base64:
            return "jpeg"
        if image_base64.startswith("iVBORw"):
            return "png"
        if image_base64.startswith("/9j/"):
            return "jpeg"
        if image_base64.startswith("UklGR"):
            return "webp"
        if image_base64.startswith("R0lGOD"):
            return "gif"
        return "jpeg"

    async def _generate_visual_summary_for_image(
        self,
        *,
        image_base64: str,
        scene_hint: str = "",
    ) -> str:
        """Phase 4：对生成的图片做一次视觉摘要（VLM），用于“配文贴图”。"""
        enabled = self.plugin.get_config("auto_selfie.enable_visual_summary", True)
        if not enabled:
            return ""
        if not image_base64:
            return ""

        try:
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config as maibot_model_config
        except Exception:
            return ""

        hint_block = f"参考场景（可能不完全正确）：{scene_hint}\n" if scene_hint else ""
        prompt = (
            "你将看到一张自拍图片。请用中文输出【视觉摘要】(1-2 句)，只描述画面里能确定的内容：\n"
            "- 人物动作/姿势/表情\n"
            "- 场景环境/地点类型（如卧室/教室/咖啡馆等）\n"
            "- 服装大类（如睡衣/校服/休闲装等）\n"
            "要求：\n"
            "1) 不要猜测画面外信息，不要编造未出现的物品\n"
            "2) 不要出现 phone/smartphone/mobile/device 等手机相关词\n"
            "3) 只输出摘要文本，不要 markdown，不要列表符号\n"
            f"{hint_block}"
        )

        try:
            vlm_request = LLMRequest(
                model_set=maibot_model_config.model_task_config.vlm,
                request_type="plugin.auto_selfie.visual_summary",
            )

            image_format = self._infer_image_format_from_base64(image_base64)
            result = await vlm_request.generate_response_for_image(
                prompt=prompt,
                image_base64=image_base64,
                image_format=image_format,
            )

            summary = ""
            if result and len(result) >= 1:
                summary = str(result[0] or "").strip()
            summary = summary.replace("\r", " ").replace("\n", " ").strip()
            summary = " ".join(summary.split())

            if summary:
                logger.info(f"{self.log_prefix} [VLM] 视觉摘要生成成功: {summary[:80]}...")
            return summary

        except Exception as e:
            logger.warning(f"{self.log_prefix} [VLM] 视觉摘要生成失败: {e}")
            return ""

    async def _is_visual_summary_consistent(self, *, planned_scene: str, visual_summary: str) -> bool:
        """可选一致性自检：判断“计划场景”与“视觉摘要”是否一致。

        - 默认关闭（auto_selfie.enable_visual_consistency_check=false）
        - 失败/异常时保守返回 True（不阻塞发送流程）
        """
        enabled = self.plugin.get_config("auto_selfie.enable_visual_consistency_check", False)
        if not enabled:
            return True

        if not planned_scene.strip() or not visual_summary.strip():
            return True

        try:
            available_models = llm_api.get_available_models()

            EXCLUDED_MODELS = {"embedding", "voice", "vlm", "lpmm_entity_extract", "lpmm_rdf_build"}
            PREFERRED_MODELS = ["replyer", "planner", "utils"]

            preferred_model_id = str(self.plugin.get_config("auto_selfie.caption_model_id", "") or "").strip()

            model_config = None
            if preferred_model_id and preferred_model_id in available_models:
                model_config = available_models[preferred_model_id]
            else:
                for model_name in PREFERRED_MODELS:
                    if model_name in available_models:
                        model_config = available_models[model_name]
                        break
                if model_config is None:
                    for model_name, cfg in available_models.items():
                        if model_name not in EXCLUDED_MODELS:
                            model_config = cfg
                            break

            if not model_config:
                return True

            prompt = (
                "你将看到【计划场景】与【视觉摘要】。请判断两者是否一致。\n"
                "一致的含义：地点类型/动作/状态不冲突（允许细节缺失）。\n"
                "如果明显冲突（比如计划说在办公室工作，但摘要显示在卧室躺着），则不一致。\n"
                "只输出 YES 或 NO，不要解释。\n\n"
                f"【计划场景】{planned_scene}\n"
                f"【视觉摘要】{visual_summary}\n"
            )

            success, content, _, _ = await llm_api.generate_with_model(
                prompt=prompt,
                model_config=model_config,
                request_type="plugin.auto_selfie.visual_consistency_check",
                temperature=0.0,
                max_tokens=5,
            )

            if not success or not content:
                return True

            ans = str(content).strip().upper()
            if ans.startswith("NO"):
                return False
            if ans.startswith("YES"):
                return True

            # 兼容可能返回中文
            if "不" in ans or "否" in ans:
                return False

            return True

        except Exception as e:
            logger.warning(f"{self.log_prefix} 一致性自检异常，跳过: {e}")
            return True

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

            if scene_llm_model and scene_llm_model in available_models:
                model_config = available_models[scene_llm_model]
                logger.debug(f"{self.log_prefix} 使用配置的 LLM 模型: {scene_llm_model}")
            else:
                # 按优先级尝试选择模型
                for model_name in PREFERRED_MODELS:
                    if model_name in available_models:
                        model_config = available_models[model_name]
                        logger.debug(f"{self.log_prefix} 使用默认 LLM 模型: {model_name}")
                        break

                # 如果首选模型都不存在，从剩余模型中选择（排除不适合的）
                if model_config is None:
                    for model_name, config in available_models.items():
                        if model_name not in EXCLUDED_MODELS:
                            model_config = config
                            logger.debug(f"{self.log_prefix} 使用备选 LLM 模型: {model_name}")
                            break

                # 如果仍然没有找到，记录警告
                if model_config is None:
                    logger.warning(
                        f"{self.log_prefix} 无法找到适合文本生成的模型，可用模型: {list(available_models.keys())}"
                    )

            if not model_config:
                logger.warning(f"{self.log_prefix} 未找到可用的 LLM 模型配置")
                return None

            # 调用 LLM 生成场景描述
            success, content, reasoning, model_name = await llm_api.generate_with_model(
                prompt=prompt,
                model_config=model_config,
                request_type="plugin.auto_selfie.scene_generate",
                temperature=0.8,  # 稍高的温度以增加创意性
                max_tokens=100,  # 场景描述不需要太长
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
        self, stream_or_id, *, description: Optional[str] = None, use_narrative_caption: bool = False
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
                chat_stream = chat_api.get_stream_by_group_id(
                    stream_id.split(":")[1] if ":" in stream_id else stream_id
                )
            except Exception:
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

            # 检查是否启用配文生成
            enable_narrative = self.plugin.get_config("auto_selfie.enable_narrative", True)

            # 如果启用配文生成且传入了 use_narrative_caption=True，使用配文生成器
            if enable_narrative and use_narrative_caption and self.caption_generator is not None:
                logger.debug(f"{self.log_prefix} 使用配文生成器生成配文")
                try:
                    # Smart 模式：使用简化的配文生成（不依赖旧版叙事系统）
                    scene_desc = description or ""
                    mood = "neutral"

                    # 选择配文类型
                    caption_type = self.caption_generator.select_caption_type(
                        scene=None,  # Smart 模式不使用旧版 NarrativeScene
                        narrative_context="",
                        current_hour=datetime.now().hour,
                    )
                    logger.debug(f"{self.log_prefix} 选择的配文类型: {caption_type.value}")

                    # 生成配文
                    ask_message = await self.caption_generator.generate_caption(
                        caption_type=caption_type,
                        scene_description=scene_desc,
                        narrative_context="",
                        image_prompt=description or "",
                        mood=mood,
                        visual_summary="",
                    )

                    logger.info(f"{self.log_prefix} 配文生成成功 (类型: {caption_type.value}): {ask_message}")

                except Exception as e:
                    import traceback

                    logger.warning(f"{self.log_prefix} 配文生成失败，回退到传统方式: {e}")
                    logger.debug(f"{self.log_prefix} 配文生成失败堆栈: {traceback.format_exc()}")
                    ask_message = ""  # 重置，使用传统方式

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
                            max_tokens=50,
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
                            "嘿嘿，来张自拍！",
                        ]
                        ask_message = random.choice(templates)

            # 3. 调用 Action 生成图片
            from .pic_action import CustomPicAction

            # 使用共享工具创建 Mock 消息对象
            mock_message = create_mock_message(chat_stream, "auto_selfie")

            # 构造 action_data
            action_data = {
                "description": "auto selfie",
                "model_id": model_id,
                "selfie_mode": True,
                "selfie_style": style,
                "size": "",
            }

            # 实例化 Action
            action_instance = CustomPicAction(
                action_data=action_data,
                action_reasoning="Auto selfie task triggered",
                cycle_timers={},
                thinking_id="auto_selfie",
                chat_stream=chat_stream,
                plugin_config=self.plugin.config,  # 传入当前插件配置
                action_message=mock_message,
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
                description=base_description,  # 使用优化后的描述
                selfie_style=style,
                free_hand_action="",
                model_id=model_id,
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
                extra_negative_prompt=neg_prompt,
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
                "auto_selfie.schedule_times",
                ["07:30", "09:00", "10:30", "12:00", "14:00", "16:00", "18:00", "20:00", "22:00"],
            )

            # 角色名称和人设自动从 MaiBot 主配置读取（在 schedule_generator 内部处理）
            weather = self.plugin.get_config("auto_selfie.weather", "晴天")
            is_holiday = self.plugin.get_config("auto_selfie.is_holiday", False)

            logger.info(f"{self.log_prefix} [Smart] 开始加载或生成日程: {today}")

            schedule = await self.schedule_generator.get_or_generate_schedule(
                date=today,
                schedule_times=schedule_times,
                weather=weather,
                is_holiday=is_holiday,
            )

            if schedule:
                with self._schedule_lock:
                    self.current_schedule = schedule
                logger.info(f"{self.log_prefix} [Smart] 日程加载成功，共 {len(schedule.entries)} 个条目")
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
        logger.info(f"{self.log_prefix} [Smart] ========== 开始处理智能日程模式 ==========")
        logger.info(f"{self.log_prefix} [Smart] 白名单流数量: {len(allowed_streams)}")
        logger.info(f"{self.log_prefix} [Smart] 模式: '生成一次，发送多次' - 只调用一次API生成图片和配文")

        # 初始化变量（确保在所有代码路径中都有定义）
        matched_time: Optional[str] = None
        fallback_key: str = f"smart_fallback_{current_date_str}"
        is_fallback_mode = False
        is_interval_trigger = False  # 是否是间隔补充触发
        current_timestamp = time.time()

        # 获取配置的时间点
        schedule_times = self.plugin.get_config(
            "auto_selfie.schedule_times",
            ["07:30", "09:00", "10:30", "12:00", "14:00", "16:00", "18:00", "20:00", "22:00"],
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
                    logger.debug(f"{self.log_prefix} [Smart] 条目 {current_entry.time_point} 已完成")
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
                f"{self.log_prefix} [Smart] 日程系统不可用，尝试使用回退机制 (仍采用'生成一次，发送多次'模式)"
            )

            # 检查是否在任何时间点的 ±2分钟窗口内
            for t_str in schedule_times:
                if ":" not in t_str or len(t_str) != 5:
                    continue
                try:
                    t_hour, t_minute = map(int, t_str.split(":"))
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
                    logger.debug(f"{self.log_prefix} [Smart-Fallback] 时间点 {matched_time} 今天已发送")
                    matched_time = None  # 重置，继续检查间隔补充
                else:
                    logger.info(f"{self.log_prefix} [Smart-Fallback] 触发回退模式发送，时间点: {matched_time}")

        # ================================================================
        # 检查间隔补充触发（日程触发之外的随机发送）
        # ================================================================
        if not current_entry and not matched_time:
            # 日程和回退都没有触发，检查间隔补充
            # v3.6.2: 暂时禁用间隔补充功能，等待后续修复"就近条目"策略导致的重复场景问题
            enable_interval = self.plugin.get_config("auto_selfie.enable_interval_supplement", False)

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
                            f"将在 {random_wait / 60:.1f} 分钟后开始检查间隔触发"
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
                                    f"(间隔: {interval_minutes}分钟, 概率: {interval_probability * 100:.0f}%)"
                                )
                        else:
                            logger.debug(
                                f"{self.log_prefix} [Smart-Interval] 概率检查未通过 "
                                f"(p={interval_probability * 100:.0f}%)"
                            )
                else:
                    logger.debug(f"{self.log_prefix} [Smart-Interval] 在时间点附近（±30分钟），跳过间隔检查")

            # 如果都没有触发，直接返回
            if not is_interval_trigger:
                logger.debug(f"{self.log_prefix} [Smart] 无触发条件，等待下一次检查")
                return

        # ================================================================
        # 关键：使用第一个流作为代表生成图片和配文（只生成一次）
        # ================================================================
        representative_stream = allowed_streams[0]

        # Phase 3：变体闭环 - 本次发送实际使用的条目/变体（发送成功后再持久化标记 is_used）
        used_schedule: Optional[DailySchedule] = None
        used_entry: Optional[ScheduleEntry] = None
        used_variation: Optional[SceneVariation] = None
        used_variation_reset_before_mark: bool = False

        def pick_variation(entry: ScheduleEntry) -> Tuple[Optional[SceneVariation], bool]:
            """选择一个场景变体，但不在这里改写 is_used。

            Returns:
                (variation, reset_before_mark)
                - variation: 选择的变体
                - reset_before_mark: 如果当前所有变体都已使用，则在发送成功后需要先 reset 再 mark
            """
            variations = entry.scene_variations or []
            if not variations:
                return None, False

            unused = [v for v in variations if not v.is_used]
            if unused:
                return random.choice(unused), False

            # 全部用过：选择一个重用，但把“reset”延后到发送成功之后
            return random.choice(variations), True

        # 确定触发类型
        trigger_type = "日程条目" if current_entry else ("回退时间点" if matched_time else "间隔补充")
        logger.info(f"{self.log_prefix} [Smart] 【生成阶段】触发类型: {trigger_type}")
        logger.info(f"{self.log_prefix} [Smart] 【生成阶段】使用代表流 {representative_stream.stream_id} 生成内容")
        logger.info(f"{self.log_prefix} [Smart] 【重要】只调用一次图片生成API，然后复用结果发送到所有流")

        # 生成图片和配文（只调用一次）
        if current_entry:
            # 使用日程条目生成
            logger.info(f"{self.log_prefix} [Smart] 使用日程条目模式生成内容")

            used_schedule = schedule
            used_entry = current_entry
            used_variation, used_variation_reset_before_mark = pick_variation(current_entry)
            if used_variation:
                logger.info(
                    f"{self.log_prefix} [Smart-Variation] 选择场景变体: "
                    f"{used_variation.variation_id} - {used_variation.description} "
                    f"(发送成功后才会标记已用)"
                )

            image_base64, caption, prompt_used = await self._generate_selfie_content_with_entry(
                representative_stream=representative_stream,
                schedule_entry=current_entry,
                schedule=schedule,
                scene_variation=used_variation,  # Phase 3：使用变体 prompt
            )
        elif is_interval_trigger:
            # 间隔补充模式：优先尝试读取当前时间对应的日程条目
            logger.info(f"{self.log_prefix} [Smart-Interval] 使用间隔补充模式生成内容")

            # 【修复 v3】使用就近条目策略 + 场景变体 + 智能场景调整
            interval_schedule = await self._ensure_daily_schedule()
            interval_entry: Optional[ScheduleEntry] = None
            time_relation: str = ""  # before/after/within
            interval_variation: Optional[SceneVariation] = None  # 选择的场景变体（发送成功后才会标记已用）
            interval_variation_reset_before_mark: bool = False

            if interval_schedule:
                # 首先尝试精确匹配（当前时间在条目的时间范围内）
                for entry in interval_schedule.entries:
                    if entry.is_time_in_range(current_time_obj):
                        interval_entry = entry
                        time_relation = "within"
                        logger.info(
                            f"{self.log_prefix} [Smart-Interval] 找到精确匹配的日程条目: "
                            f"{entry.time_point} - {entry.activity_description}"
                        )
                        break

                # 如果没有精确匹配，使用就近条目策略
                if not interval_entry:
                    interval_entry, time_relation = interval_schedule.get_closest_entry(current_time_obj)
                    if interval_entry:
                        logger.info(
                            f"{self.log_prefix} [Smart-Interval] 使用就近日程条目: "
                            f"{interval_entry.time_point} - {interval_entry.activity_description} "
                            f"(时间关系: {time_relation})"
                        )

                # Phase 3：尝试选择场景变体以避免重复（发送成功后再持久化标记 is_used）
                if interval_entry and interval_entry.scene_variations:
                    interval_variation, interval_variation_reset_before_mark = pick_variation(interval_entry)

                    if interval_variation:
                        logger.info(
                            f"{self.log_prefix} [Smart-Interval-Variation] 选择场景变体: "
                            f"{interval_variation.variation_id} - {interval_variation.description} "
                            f"(发送成功后才会标记已用)"
                        )

                    # 记录变体使用情况
                    logger.debug(
                        f"{self.log_prefix} [Smart-Interval-Variation] 条目 {interval_entry.time_point} "
                        f"变体使用次数: {interval_entry.interval_use_count}, "
                        f"可用变体: {len(interval_entry.scene_variations)}"
                    )
                elif interval_entry:
                    # 没有预定义变体，但条目存在
                    logger.info(f"{self.log_prefix} [Smart-Interval] 条目无预定义变体，将使用 LLM 场景调整以增加变化")

            if interval_entry:
                # 有匹配的日程条目，使用日程驱动方式
                # 传递时间关系和变体用于智能调整
                logger.info(
                    f"{self.log_prefix} [Smart-Interval] 使用日程条目驱动场景: "
                    f"地点={interval_entry.location}, 服装={interval_entry.outfit}, "
                    f"时间关系={time_relation}, 变体={'是' if interval_variation else '否'}"
                )
                used_schedule = interval_schedule
                used_entry = interval_entry
                used_variation = interval_variation
                used_variation_reset_before_mark = interval_variation_reset_before_mark

                image_base64, caption, prompt_used = await self._generate_selfie_content_with_entry(
                    representative_stream=representative_stream,
                    schedule_entry=interval_entry,
                    schedule=interval_schedule,
                    time_relation=time_relation,
                    scene_variation=interval_variation,  # Phase 3：使用变体 prompt
                )
            else:
                # 没有匹配的日程条目（日程为空），回退到 LLM 生成场景
                logger.info(f"{self.log_prefix} [Smart-Interval] 无日程条目，使用 LLM 生成场景")
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
        logger.info(f"{self.log_prefix} [Smart] 【发送阶段】图片生成成功，开始发送到 {len(allowed_streams)} 个流")
        logger.info(f"{self.log_prefix} [Smart] 【重要】以下所有流将收到相同的图片和配文")

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

        logger.info(f"{self.log_prefix} [Smart] ========== 自拍发送完成 ==========")
        logger.info(f"{self.log_prefix} [Smart] 发送结果: 成功 {success_count}/{len(allowed_streams)} 个流")

        # 标记完成并保存（仅在至少 1 个流发送成功时进行持久化，避免失败也消耗条目/变体）
        send_any_success = success_count > 0

        if current_entry and schedule:
            if not send_any_success:
                logger.warning(
                    f"{self.log_prefix} [Smart] 本次发送全部失败，"
                    f"不标记条目完成也不标记变体已用: {current_entry.time_point}"
                )
                return

            # 正常模式：标记日程条目完成
            schedule.mark_entry_completed(current_entry.time_point, caption)

            # Phase 3：发送成功后再标记变体已用并持久化
            if used_entry and used_variation:
                if used_variation_reset_before_mark:
                    used_entry.reset_variations()
                used_entry.mark_variation_used(used_variation.variation_id)

            # Phase 5：发送成功后更新叙事状态（用于后续配文承上启下）
            try:
                schedule.narrative_state.update_after_send(
                    entry=current_entry,
                    variation=used_variation,
                    is_interval=False,
                )
            except Exception as e:
                logger.warning(f"{self.log_prefix} [Smart] 更新叙事状态失败，跳过: {e}")

            if self.schedule_generator is not None:
                schedule.save_to_file(self.schedule_generator.get_schedule_file_path(schedule.date))

            # 更新缓存
            with self._schedule_lock:
                self.current_schedule = schedule
            logger.debug(f"{self.log_prefix} [Smart] 日程条目已标记完成: {current_entry.time_point}")

        elif is_fallback_mode and matched_time:
            if not send_any_success:
                logger.warning(f"{self.log_prefix} [Smart-Fallback] 本次发送全部失败，不记录已发送状态: {matched_time}")
                return

            # 回退模式：记录发送状态
            if fallback_key not in self.last_send_dates:
                self.last_send_dates[fallback_key] = {}
            self.last_send_dates[fallback_key][matched_time] = current_date_str
            self._save_state()
            logger.debug(f"{self.log_prefix} [Smart-Fallback] 发送状态已保存: {matched_time}")

        elif is_interval_trigger:
            if not send_any_success:
                logger.warning(
                    f"{self.log_prefix} [Smart-Interval] 本次发送全部失败，不更新间隔计时器，也不持久化变体使用"
                )
                return

            # 间隔补充模式：更新间隔计时器
            interval_key = "smart_interval_global"
            self.last_send_time[interval_key] = current_timestamp
            self._save_state()

            # Phase 3：发送成功后再标记变体已用并持久化
            if used_schedule and used_entry and self.schedule_generator is not None:
                if used_variation:
                    if used_variation_reset_before_mark:
                        used_entry.reset_variations()
                    used_entry.mark_variation_used(used_variation.variation_id)

                # 记录一次间隔补充使用（用于后续调参/策略）
                used_entry.record_interval_use()

                # Phase 5：间隔补充发送也更新叙事状态
                try:
                    used_schedule.narrative_state.update_after_send(
                        entry=used_entry,
                        variation=used_variation,
                        is_interval=True,
                    )
                except Exception as e:
                    logger.warning(f"{self.log_prefix} [Smart-Interval] 更新叙事状态失败，跳过: {e}")

                used_schedule.save_to_file(self.schedule_generator.get_schedule_file_path(used_schedule.date))

                with self._schedule_lock:
                    self.current_schedule = used_schedule

            logger.debug(f"{self.log_prefix} [Smart-Interval] 间隔计时器已更新")

    async def _generate_selfie_content_with_entry(
        self,
        representative_stream,
        schedule_entry: ScheduleEntry,
        schedule: Optional[DailySchedule] = None,
        time_relation: str = "within",
        scene_variation: Optional[SceneVariation] = None,
    ) -> Tuple[Optional[str], str, str]:
        """使用日程条目生成自拍图片和配文

        Args:
            representative_stream: 代表流（用于初始化 Action）
            schedule_entry: 日程条目
            schedule: 当天日程（可选）。用于 Phase 5：提供叙事上下文（承上启下）
            time_relation: 时间关系，用于智能调整配文风格
                - "within": 在条目时间范围内（默认，正常场景）
                - "before": 当前时间在条目时间之前（准备中、期待中）
                - "after": 当前时间在条目时间之后（结束后、休息中）
            scene_variation: 可选的场景变体，用于避免重复场景

        Returns:
            Tuple[图片base64, 配文, 使用的prompt]
        """
        from .pic_action import CustomPicAction

        chat_stream = representative_stream

        # 日志记录
        if scene_variation:
            logger.info(
                f"{self.log_prefix} [Smart] 使用日程条目+场景变体生成自拍内容: "
                f"变体={scene_variation.variation_id} ({scene_variation.description})"
            )
        else:
            logger.info(f"{self.log_prefix} [Smart] 使用日程条目生成自拍内容")

        try:
            # 1. 获取配置
            style = self.plugin.get_config("auto_selfie.selfie_style", "standard")
            model_id = self.plugin.get_config("auto_selfie.model_id", "model1")

            # 2. 使用 SceneActionGenerator 生成配文上下文
            scene_generator = SceneActionGenerator(self.plugin)
            caption_context = scene_generator.create_caption_context(schedule_entry)

            # 2.1 如果有变体，使用变体的配文主题
            variation_caption_theme: Optional[str] = None
            if scene_variation and scene_variation.caption_theme:
                variation_caption_theme = scene_variation.caption_theme
                logger.debug(f"{self.log_prefix} [Smart-Variation] 使用变体配文主题: {variation_caption_theme}")

            # 3. 使用共享工具创建 Mock 消息对象
            mock_message = create_mock_message(chat_stream, "smart_selfie")

            action_data = {
                "description": "smart selfie",
                "model_id": model_id,
                "selfie_mode": True,
                "selfie_style": style,
                "size": "",
            }

            action_instance = CustomPicAction(
                action_data=action_data,
                action_reasoning="Smart schedule selfie triggered",
                cycle_timers={},
                thinking_id="smart_selfie",
                chat_stream=chat_stream,
                plugin_config=self.plugin.config,
                action_message=mock_message,
            )

            # 5. 确定场景提示词
            adjusted_scene_prompt: Optional[str] = None

            # 5.1 优先使用变体的场景提示词
            if scene_variation:
                # 使用 SceneActionGenerator 生成完整变体 prompt（并对手机相关词做兜底）
                adjusted_scene_prompt = scene_generator.convert_to_sd_prompt(
                    schedule_entry,
                    selfie_style=style,
                    scene_variation=scene_variation,
                )
                logger.info(f"{self.log_prefix} [Smart-Variation] 使用变体场景提示词: {adjusted_scene_prompt[:100]}...")
            elif time_relation != "within":
                # 5.2 没有变体但时间关系不是 within，使用 LLM 调整场景
                logger.info(
                    f"{self.log_prefix} [Smart] 检测到时间关系为 '{time_relation}'，尝试使用 LLM 调整场景以增加变化"
                )
                adjusted_scene_prompt = await self._adjust_scene_for_time_relation(
                    schedule_entry=schedule_entry,
                    time_relation=time_relation,
                )
                if adjusted_scene_prompt:
                    logger.info(f"{self.log_prefix} [Smart] 场景调整成功，将使用调整后的场景生成图片")
                else:
                    logger.info(f"{self.log_prefix} [Smart] 场景调整失败或返回空，继续使用原始日程场景")

            # 6. 使用场景驱动方式生成提示词
            # 如果有调整后的场景描述，将其作为额外描述传入
            prompt = action_instance._process_selfie_prompt(
                description=adjusted_scene_prompt or "",  # 使用调整后的场景或空描述
                selfie_style=style,
                free_hand_action="",
                model_id=model_id,
                schedule_entry=schedule_entry,  # 传入日程条目（服装、地点等仍从这里读取）
            )

            # 7. 获取负面提示词
            neg_prompt = scene_generator.get_negative_prompt_for_style(style)

            # 8. 获取参考图
            ref_image = action_instance._get_selfie_reference_image()

            # 9. 执行图片生成
            image_base64 = await action_instance._generate_image_only(
                description=prompt,
                model_id=model_id,
                size="",
                strength=0.6,
                input_image_base64=ref_image,
                extra_negative_prompt=neg_prompt,
            )

            if not image_base64:
                return None, "", ""

            # Phase 4：VLM 视觉摘要 -> 配文贴图
            scene_hint = schedule_entry.activity_description
            if scene_variation and scene_variation.description:
                scene_hint = f"{scene_hint}（变体：{scene_variation.description}）"

            visual_summary = await self._generate_visual_summary_for_image(
                image_base64=image_base64,
                scene_hint=scene_hint,
            )

            scene_desc_override: Optional[str] = None
            if visual_summary:
                try:
                    is_consistent = await self._is_visual_summary_consistent(
                        planned_scene=scene_hint,
                        visual_summary=visual_summary,
                    )
                    if not is_consistent:
                        scene_desc_override = visual_summary
                except Exception as e:
                    logger.warning(f"{self.log_prefix} 一致性自检失败，跳过: {e}")

            # 3. 生成配文（基于视觉摘要；失败则回退）
            caption = await self._generate_caption_for_entry(
                schedule_entry,
                caption_context,
                time_relation,
                schedule=schedule,
                scene_variation=scene_variation,
                scene_description_override=scene_desc_override,
                visual_summary=visual_summary,
            )

            return image_base64, caption, prompt

        except Exception as e:
            logger.error(f"{self.log_prefix} [Smart] 生成自拍内容失败: {e}", exc_info=True)
            return None, "", ""

    async def _adjust_scene_for_time_relation(
        self,
        schedule_entry: ScheduleEntry,
        time_relation: str,
    ) -> Optional[str]:
        """使用 LLM 根据时间关系动态调整场景描述

        当间隔补充触发时（time_relation 为 "before" 或 "after"），
        调用 LLM 生成一个与原始日程条目相关但有所变化的场景描述。

        Args:
            schedule_entry: 原始日程条目
            time_relation: 时间关系
                - "within": 在条目时间范围内，返回 None（使用原始场景）
                - "before": 当前时间在条目时间之前
                - "after": 当前时间在条目时间之后

        Returns:
            调整后的场景描述（SD 提示词格式），如果无需调整返回 None
        """
        # 如果是 within，直接返回 None，使用原始场景
        if time_relation == "within":
            logger.debug(f"{self.log_prefix} [SceneAdjust] 时间关系为 within，使用原始场景")
            return None

        try:
            # 获取当前时间信息
            now = datetime.now()
            time_str = now.strftime("%H:%M")

            # 获取 bot 名称
            try:
                bot_name = global_config.bot.nickname
                if not bot_name:
                    bot_name = "角色"
            except Exception:
                bot_name = "角色"

            # 根据时间关系构建不同的提示词
            if time_relation == "before":
                relation_desc = "即将开始"
                variation_hint = (
                    "场景应该体现'准备中'或'期待中'的状态，比如：\n"
                    "- 整理装备、查看时间\n"
                    "- 换衣服中、化妆中\n"
                    "- 收拾东西、准备出门\n"
                    "- 看着窗外、等待中"
                )
            else:  # after
                relation_desc = "刚刚结束"
                variation_hint = (
                    "场景应该体现'结束后休息'或'过渡'的状态，比如：\n"
                    "- 放下道具、换个姿势\n"
                    "- 躺着休息、伸懒腰\n"
                    "- 喝水、吃零食\n"
                    "- 放空发呆、休息眼睛"
                )

            # 构建提示词
            prompt = f"""Current time: {time_str}
Character: {bot_name}
Original scheduled activity: {schedule_entry.activity_description}
Original outfit: {schedule_entry.outfit}
Original location: {schedule_entry.location}
Time relation: The scheduled activity has {relation_desc}

Task: Generate a variation of the selfie scene that reflects this time relation.

{variation_hint}

Requirements:
1. Keep the same location ({schedule_entry.location}) and general outfit theme
2. Change the pose and action to reflect the '{relation_desc}' state
3. Output format: English SD prompt tags, comma-separated
4. Include: pose variation, hand action, expression, any small detail changes
5. Keep it concise (50-80 words)
6. Do NOT include character appearance (hair, eyes, etc.)
7. Do NOT repeat the full outfit description, just mention small variations if any

Example for "after" state of "宅家放松 (playing Switch)":
"lying on couch, arms stretched, yawning, Switch controller on lap, eyes half-closed, relaxed expression, cozy, messy hair"

Example for "before" state of "下午茶 (cafe time)":
"checking mirror, adjusting hair, holding bag, standing, anticipating expression, ready to go out"

Now generate for the '{relation_desc}' state of "{schedule_entry.activity_description}":"""

            # 获取可用模型
            available_models = llm_api.get_available_models()

            if not available_models:
                logger.warning(f"{self.log_prefix} [SceneAdjust] 无可用的 LLM 模型")
                return None

            # 选择模型
            EXCLUDED_MODELS = {"embedding", "voice", "vlm", "lpmm_entity_extract", "lpmm_rdf_build"}
            PREFERRED_MODELS = ["replyer", "planner", "utils"]

            model_config = None

            for model_name in PREFERRED_MODELS:
                if model_name in available_models:
                    model_config = available_models[model_name]
                    break

            if model_config is None:
                for model_name, config in available_models.items():
                    if model_name not in EXCLUDED_MODELS:
                        model_config = config
                        break

            if not model_config:
                logger.warning(f"{self.log_prefix} [SceneAdjust] 未找到可用的 LLM 模型配置")
                return None

            # 调用 LLM
            success, content, _, model_name = await llm_api.generate_with_model(
                prompt=prompt,
                model_config=model_config,
                request_type="plugin.auto_selfie.scene_adjust",
                temperature=0.9,  # 较高温度以增加变化
                max_tokens=150,
            )

            if success and content:
                # 清理返回结果
                adjusted_scene = content.strip().strip('"').strip("'").strip()
                # 移除可能的前缀解释
                if ":" in adjusted_scene and len(adjusted_scene.split(":")[0]) < 30:
                    adjusted_scene = adjusted_scene.split(":", 1)[-1].strip()

                logger.info(
                    f"{self.log_prefix} [SceneAdjust] LLM 场景调整成功 (时间关系: {time_relation}, 模型: {model_name})"
                )
                logger.info(f"{self.log_prefix} [SceneAdjust] 调整后场景: {adjusted_scene[:100]}...")
                return adjusted_scene
            else:
                logger.warning(f"{self.log_prefix} [SceneAdjust] LLM 场景调整失败: {content}")
                return None

        except Exception as e:
            logger.error(f"{self.log_prefix} [SceneAdjust] 场景调整出错: {e}", exc_info=True)
            return None

    async def _generate_caption_for_entry(
        self,
        schedule_entry: ScheduleEntry,
        caption_context: Dict[str, str],
        time_relation: str = "within",
        schedule: Optional[DailySchedule] = None,
        scene_variation: Optional[SceneVariation] = None,
        scene_description_override: Optional[str] = None,
        visual_summary: str = "",
    ) -> str:
        """为日程条目生成配文

        支持时间关系感知的智能配文生成：
        - within: 正在进行的场景，使用正常配文风格
        - before: 在日程时间之前，使用"准备中/期待中"的配文风格
        - after: 在日程时间之后，使用"刚结束/休息中"的配文风格

        如果传入了 scene_variation，将优先使用变体的配文主题和情绪。

        Args:
            schedule_entry: 日程条目
            caption_context: 配文上下文
            time_relation: 时间关系 ("within", "before", "after")
            scene_variation: 可选的场景变体

        Returns:
            生成的配文
        """
        # 确定场景描述和情绪
        adjusted_scene: str
        adjusted_mood: str
        caption_theme: str

        # 如果有场景变体，优先使用变体的信息
        if scene_variation:
            adjusted_scene = scene_variation.description or schedule_entry.activity_description
            adjusted_mood = scene_variation.mood or schedule_entry.mood
            caption_theme = scene_variation.caption_theme or schedule_entry.suggested_caption_theme
            logger.info(
                f"{self.log_prefix} [Smart-Variation] 使用变体配文信息: "
                f"场景={adjusted_scene}, 情绪={adjusted_mood}, 主题={caption_theme}"
            )
        else:
            # 根据时间关系调整场景描述
            adjusted_scene = schedule_entry.activity_description
            adjusted_mood = schedule_entry.mood
            caption_theme = schedule_entry.suggested_caption_theme

            if time_relation == "before":
                # 在日程时间之前：准备中、期待中的状态
                adjusted_scene = f"准备{schedule_entry.activity_description}"
                logger.info(
                    f"{self.log_prefix} [Smart-TimeRelation] 时间在日程之前，调整场景为'准备中'风格: {adjusted_scene}"
                )
                # 可以调整情绪为更轻松/期待的
                if adjusted_mood in ["neutral", "relaxed"]:
                    adjusted_mood = "anticipating"
            elif time_relation == "after":
                # 在日程时间之后：结束后、休息中的状态
                adjusted_scene = f"{schedule_entry.activity_description}结束后休息中"
                logger.info(
                    f"{self.log_prefix} [Smart-TimeRelation] 时间在日程之后，调整场景为'休息中'风格: {adjusted_scene}"
                )
                # 可以调整情绪为更放松的
                if adjusted_mood in ["neutral", "focused"]:
                    adjusted_mood = "relaxed"
            else:
                logger.debug(f"{self.log_prefix} [Smart-TimeRelation] 时间在日程范围内，使用原始场景: {adjusted_scene}")

        # 视觉摘要优先：当出现明显不一致时，上层会传入 scene_description_override
        scene_desc_for_caption = adjusted_scene
        if scene_description_override and scene_description_override.strip():
            scene_desc_for_caption = scene_description_override.strip()

        # 首先尝试使用叙事配文系统
        if self.caption_generator is not None:
            try:
                # 将 caption_type 字符串转换为 CaptionType 枚举
                caption_type_str = schedule_entry.caption_type.upper()
                try:
                    caption_type = CaptionType(caption_type_str.lower())
                except ValueError:
                    caption_type = CaptionType.SHARE

                # 根据时间关系调整配文类型（仅在没有变体时）
                if not scene_variation:
                    if time_relation == "before":
                        # 准备阶段更适合使用分享或独白类型
                        if caption_type not in [CaptionType.SHARE, CaptionType.MONOLOGUE]:
                            caption_type = CaptionType.SHARE
                    elif time_relation == "after":
                        # 结束后更适合使用独白或分享类型
                        if caption_type not in [CaptionType.MONOLOGUE, CaptionType.SHARE]:
                            caption_type = CaptionType.MONOLOGUE

                narrative_context = caption_context.get("activity_detail", "")
                if schedule is not None:
                    try:
                        narrative_context = schedule.narrative_state.get_context_for_caption()
                    except Exception:
                        # 保守回退：使用旧的“已完成条目摘要”接口
                        narrative_context = schedule.get_narrative_context(max_entries=3)

                caption = await self.caption_generator.generate_caption(
                    caption_type=caption_type,
                    scene_description=scene_desc_for_caption,
                    narrative_context=narrative_context,
                    image_prompt=caption_theme,  # 使用（可能来自变体的）配文主题
                    mood=adjusted_mood,  # 使用调整后的情绪
                    visual_summary=visual_summary,
                )

                if caption:
                    variation_info = f", 变体={scene_variation.variation_id}" if scene_variation else ""
                    logger.info(
                        f"{self.log_prefix} [Smart] 配文生成成功 "
                        f"(类型: {caption_type.value}, 时间关系: {time_relation}{variation_info})"
                    )
                    return caption

            except Exception as e:
                logger.warning(f"{self.log_prefix} [Smart] 叙事配文生成失败: {e}")

        # 回退到传统方式（使用调整后的场景描述）
        fallback_desc = adjusted_scene
        if scene_description_override and scene_description_override.strip():
            fallback_desc = scene_description_override.strip()
        if visual_summary and visual_summary.strip():
            fallback_desc = f"{fallback_desc}（图像摘要：{visual_summary.strip()}）"
        return await self._generate_traditional_caption(fallback_desc)
