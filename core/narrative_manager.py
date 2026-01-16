"""
日常叙事管理器模块

负责管理每日叙事状态、剧本选择、场景匹配等功能。
状态持久化使用 JSON 文件存储在插件目录下。
"""

import json
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.common.logger import get_logger

from .selfie_models import (
    CaptionType,
    DailyNarrativeState,
    DEFAULT_NARRATIVE_SCRIPTS,
    NarrativeScene,
)

logger = get_logger("NarrativeManager")


class NarrativeManager:
    """日常叙事管理器

    负责管理每日叙事状态、剧本选择、场景匹配等功能。
    状态持久化使用 JSON 文件存储在插件目录下。
    """

    def __init__(self, plugin_instance: Any, state_file_path: str):
        """初始化管理器

        Args:
            plugin_instance: 插件实例，用于读取配置
            state_file_path: 状态文件路径
        """
        self.plugin = plugin_instance
        self.state_file_path = state_file_path
        self._lock = threading.RLock()  # 使用可重入锁，避免嵌套调用时死锁
        self._state: Optional[DailyNarrativeState] = None

        # 加载或创建状态
        self._state = self.load_state()
        logger.info(f"NarrativeManager 初始化完成，状态文件: {state_file_path}")

    # ==================== 状态管理 ====================

    def load_state(self) -> DailyNarrativeState:
        """加载当日叙事状态

        - 如果状态文件不存在或日期不是今天，创建新状态
        - 支持跨日自动重置

        Returns:
            当日的叙事状态
        """
        with self._lock:
            today = self._get_today_str()

            try:
                with open(self.state_file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                state = DailyNarrativeState.from_dict(data)

                # 检查是否需要重置（日期不是今天）
                if state.date != today:
                    logger.info(f"检测到跨日，从 {state.date} 重置到 {today}")
                    return self._create_new_state(today)

                logger.debug(f"加载已有状态，日期: {state.date}")
                return state

            except FileNotFoundError:
                logger.info("状态文件不存在，创建新状态")
                return self._create_new_state(today)

            except json.JSONDecodeError as e:
                logger.warning(f"状态文件解析失败: {e}，创建新状态")
                return self._create_new_state(today)

            except Exception as e:
                logger.error(f"加载状态失败: {e}，创建新状态")
                return self._create_new_state(today)

    def save_state(self) -> None:
        """保存状态到文件"""
        with self._lock:
            if self._state is None:
                logger.warning("状态为空，无法保存")
                return

            try:
                with open(self.state_file_path, "w", encoding="utf-8") as f:
                    json.dump(self._state.to_dict(), f, ensure_ascii=False, indent=2)
                logger.debug("状态保存成功")

            except Exception as e:
                logger.error(f"保存状态失败: {e}")

    def reset_daily_state(self) -> DailyNarrativeState:
        """重置每日状态（新的一天开始）

        Returns:
            新创建的叙事状态
        """
        today = self._get_today_str()
        self._state = self._create_new_state(today)
        self.save_state()
        logger.info(f"每日状态已重置，日期: {today}")
        return self._state

    def _create_new_state(self, date: str) -> DailyNarrativeState:
        """创建新的每日状态

        Args:
            date: 日期字符串 YYYY-MM-DD

        Returns:
            新创建的叙事状态
        """
        script_id = self.select_daily_script()
        state = DailyNarrativeState(
            date=date,
            script_id=script_id,
            completed_scenes=[],
            context_memory=[],
            current_mood="neutral",
            last_caption="",
            last_scene_time="",
        )
        self._state = state
        self.save_state()
        return state

    # ==================== 剧本管理 ====================

    def get_script(self, script_id: Optional[str] = None) -> List[NarrativeScene]:
        """获取剧本场景列表

        Args:
            script_id: 剧本ID，None则返回当日剧本

        Returns:
            场景列表
        """
        if script_id is None:
            if self._state is not None:
                script_id = self._state.script_id
            else:
                script_id = "default"

        # 从默认剧本中获取
        if script_id in DEFAULT_NARRATIVE_SCRIPTS:
            return DEFAULT_NARRATIVE_SCRIPTS[script_id]

        # 如果找不到指定剧本，返回默认剧本
        logger.warning(f"剧本 {script_id} 不存在，使用默认剧本")
        return DEFAULT_NARRATIVE_SCRIPTS.get("default", [])

    def select_daily_script(self) -> str:
        """选择当日剧本

        根据星期几、是否节假日等因素选择合适的剧本
        周末用 "weekend" 剧本，工作日用 "default" 剧本

        Returns:
            剧本ID
        """
        if self._is_weekend():
            script_id = "weekend"
            logger.info("今天是周末，选择 weekend 剧本")
        else:
            script_id = "default"
            logger.info("今天是工作日，选择 default 剧本")

        return script_id

    # ==================== 场景匹配 ====================

    def get_current_scene(self, current_time: Optional[str] = None) -> Optional[NarrativeScene]:
        """获取当前时间应该触发的场景

        Args:
            current_time: 当前时间 "HH:MM"，None则使用系统时间

        Returns:
            匹配的场景，或 None（如果不在任何场景时间范围内或已完成）
        """
        if current_time is None:
            now = datetime.now()
            current_time = now.strftime("%H:%M")

        # 确保状态是今天的
        if self._state is None or self._state.date != self._get_today_str():
            self._state = self.load_state()

        # 获取当日剧本的场景列表
        scenes = self.get_script()

        # 查找适用的场景
        for scene in scenes:
            if self.is_scene_applicable(scene, current_time):
                # 检查场景是否已完成
                if self._state.is_scene_completed(scene.scene_id):
                    logger.debug(f"场景 {scene.scene_id} 已完成，跳过")
                    continue

                # 检查叙事连贯性
                if not self.check_narrative_continuity(scene):
                    logger.debug(f"场景 {scene.scene_id} 不满足连贯性要求，跳过")
                    continue

                logger.info(f"匹配到场景: {scene.scene_id} ({scene.description})")
                return scene

        logger.debug(f"当前时间 {current_time} 没有匹配的场景")
        return None

    def is_scene_applicable(self, scene: NarrativeScene, current_time: str) -> bool:
        """检查场景是否适用于当前时间

        Args:
            scene: 要检查的场景
            current_time: 当前时间 "HH:MM"

        Returns:
            如果场景适用于当前时间返回 True
        """
        return self._is_time_in_range(current_time, scene.time_start, scene.time_end)

    def mark_scene_completed(self, scene_id: str, caption: str) -> None:
        """标记场景已完成

        Args:
            scene_id: 场景ID
            caption: 该场景的配文内容
        """
        if self._state is None:
            logger.warning("状态为空，无法标记场景完成")
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._state.add_completed_scene(scene_id, caption, timestamp)
        self.save_state()
        logger.info(f"场景 {scene_id} 已标记完成，时间: {timestamp}")

    # ==================== 上下文管理 ====================

    def get_narrative_context(self, max_entries: int = 5) -> str:
        """获取叙事上下文（用于 LLM 生成配文）

        Args:
            max_entries: 最多返回的条目数

        Returns:
            格式化的上下文字符串，包含最近的场景和配文
        """
        if self._state is None or not self._state.context_memory:
            return "今天还没有发过自拍。"

        # 获取最近的记忆条目
        recent_entries = self._state.context_memory[-max_entries:]

        context_parts = []
        context_parts.append(f"今天是 {self._state.date}，")
        context_parts.append(f"当前情绪: {self._state.current_mood}")
        context_parts.append("\n今天的自拍记录:")

        for entry in recent_entries:
            scene_id = entry.get("scene_id", "unknown")
            caption = entry.get("caption", "")
            timestamp = entry.get("timestamp", "")

            # 获取场景描述
            scene_desc = self._get_scene_description(scene_id)
            time_part = timestamp.split(" ")[1] if " " in timestamp else timestamp

            context_parts.append(f"- [{time_part}] {scene_desc}: \"{caption}\"")

        return "\n".join(context_parts)

    def get_last_caption(self) -> str:
        """获取上一条配文

        Returns:
            上一条配文内容，如果没有则返回空字符串
        """
        if self._state is None:
            return ""
        return self._state.last_caption

    def update_mood(self, new_mood: str) -> None:
        """更新当前情绪状态

        Args:
            new_mood: 新的情绪状态
        """
        if self._state is None:
            logger.warning("状态为空，无法更新情绪")
            return

        old_mood = self._state.current_mood
        self._state.current_mood = new_mood
        self.save_state()
        logger.info(f"情绪状态已更新: {old_mood} -> {new_mood}")

    def _get_scene_description(self, scene_id: str) -> str:
        """获取场景描述

        Args:
            scene_id: 场景ID

        Returns:
            场景描述，如果找不到返回场景ID
        """
        # 遍历所有剧本查找场景
        for script_scenes in DEFAULT_NARRATIVE_SCRIPTS.values():
            for scene in script_scenes:
                if scene.scene_id == scene_id:
                    return scene.description

        return scene_id

    # ==================== 连贯性检查 ====================

    def check_narrative_continuity(self, scene: NarrativeScene) -> bool:
        """检查叙事连贯性

        如果场景有前置场景要求，检查前置场景是否已完成

        Args:
            scene: 要检查的场景

        Returns:
            如果满足连贯性要求返回 True
        """
        # 如果没有前置场景要求，直接通过
        if scene.prev_scene_ids is None or len(scene.prev_scene_ids) == 0:
            return True

        if self._state is None:
            return False

        # 检查是否有任意一个前置场景已完成
        for prev_scene_id in scene.prev_scene_ids:
            if self._state.is_scene_completed(prev_scene_id):
                return True

        # 如果是当天第一个场景，且有前置要求，检查是否还有时间完成前置
        # 这里简化处理：如果没有任何场景完成，允许触发任何场景
        if len(self._state.completed_scenes) == 0:
            logger.debug(f"今天还没有完成任何场景，允许触发 {scene.scene_id}")
            return True

        return False

    def get_transition_hint(self, from_scene_id: str, to_scene_id: str) -> str:
        """获取场景转换提示（用于生成过渡性配文）

        Args:
            from_scene_id: 源场景ID
            to_scene_id: 目标场景ID

        Returns:
            转换提示文本
        """
        from_desc = self._get_scene_description(from_scene_id)
        to_desc = self._get_scene_description(to_scene_id)

        # 生成自然的转换提示
        transition_hints = {
            ("morning_wakeup", "morning_breakfast"): "从睡醒到吃早餐，开始新的一天",
            ("morning_breakfast", "morning_ready"): "吃完早餐准备出门",
            ("morning_ready", "noon_work"): "出门后开始工作或学习",
            ("noon_work", "lunch_time"): "工作告一段落，该吃午饭了",
            ("lunch_time", "afternoon_break"): "午餐后稍作休息",
            ("afternoon_break", "afternoon_activity"): "休息够了开始下午的活动",
            ("afternoon_activity", "evening_dinner"): "活动结束，回家吃晚餐",
            ("evening_dinner", "evening_relax"): "晚餐后放松时光",
            ("evening_relax", "night_routine"): "开始准备睡觉",
            ("night_routine", "night_sleep"): "洗漱完毕，准备入睡",
            # 周末剧本转换
            ("weekend_morning", "weekend_brunch"): "周末懒觉后享用早午餐",
            ("weekend_brunch", "weekend_outing"): "吃完出门玩耍",
            ("weekend_outing", "weekend_evening"): "玩累了回家休息",
        }

        # 查找预定义的转换提示
        key = (from_scene_id, to_scene_id)
        if key in transition_hints:
            return transition_hints[key]

        # 生成通用转换提示
        return f"从「{from_desc}」过渡到「{to_desc}」"

    # ==================== 辅助方法 ====================

    def _parse_time(self, time_str: str) -> Tuple[int, int]:
        """解析时间字符串为 (hour, minute) 元组

        Args:
            time_str: 时间字符串，格式 "HH:MM"

        Returns:
            (小时, 分钟) 元组
        """
        parts = time_str.split(":")
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0
        return (hour, minute)

    def _is_time_in_range(self, current: str, start: str, end: str) -> bool:
        """检查时间是否在范围内

        支持跨午夜的时间范围（如 22:00 - 01:00）

        Args:
            current: 当前时间 "HH:MM"
            start: 开始时间 "HH:MM"
            end: 结束时间 "HH:MM"

        Returns:
            如果当前时间在范围内返回 True
        """
        current_tuple = self._parse_time(current)
        start_tuple = self._parse_time(start)
        end_tuple = self._parse_time(end)

        # 将时间转换为分钟数便于比较
        current_mins = current_tuple[0] * 60 + current_tuple[1]
        start_mins = start_tuple[0] * 60 + start_tuple[1]
        end_mins = end_tuple[0] * 60 + end_tuple[1]

        # 处理跨午夜的情况
        if end_mins < start_mins:
            # 跨午夜：检查是否在 start 到 24:00 或 00:00 到 end 范围内
            return current_mins >= start_mins or current_mins <= end_mins
        else:
            # 正常情况：检查是否在 start 到 end 范围内
            return start_mins <= current_mins <= end_mins

    def _get_today_str(self) -> str:
        """获取今日日期字符串 YYYY-MM-DD

        Returns:
            今日日期字符串
        """
        return datetime.now().strftime("%Y-%m-%d")

    def _is_weekend(self) -> bool:
        """判断今天是否是周末

        Returns:
            如果是周六或周日返回 True
        """
        # weekday() 返回 0-6，其中 0 是周一，5 和 6 是周末
        return datetime.now().weekday() >= 5

    # ==================== 属性访问 ====================

    @property
    def state(self) -> Optional[DailyNarrativeState]:
        """获取当前状态

        Returns:
            当前的叙事状态
        """
        return self._state

    @property
    def current_script_id(self) -> str:
        """获取当前剧本ID

        Returns:
            当前剧本ID，如果状态为空返回 "default"
        """
        if self._state is not None:
            return self._state.script_id
        return "default"

    @property
    def completed_scene_count(self) -> int:
        """获取已完成场景数量

        Returns:
            已完成的场景数量
        """
        if self._state is not None:
            return len(self._state.completed_scenes)
        return 0
