"""
动态日程系统数据模型模块

包含活动类型枚举、日程条目和每日日程等数据类型，
用于 LLM 动态生成日程的新架构。

v2.0 更新：
- 新增 SceneVariation：场景变体，用于同一时间段内的多次发送
- 新增 DailyNarrativeState：叙事状态追踪，保持一天的连续性
- ScheduleEntry 支持场景变体列表
- DailySchedule 集成叙事状态
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from src.common.logger import get_logger

logger = get_logger("ScheduleModels")


class ActivityType(Enum):
    """
    活动类型枚举

    定义了常见的日常活动类型，用于分类日程条目。
    """

    SLEEPING = "sleeping"  # 睡觉
    WAKING_UP = "waking_up"  # 起床
    EATING = "eating"  # 用餐
    WORKING = "working"  # 工作
    STUDYING = "studying"  # 学习
    EXERCISING = "exercising"  # 运动
    RELAXING = "relaxing"  # 休闲放松
    SOCIALIZING = "socializing"  # 社交
    COMMUTING = "commuting"  # 通勤
    HOBBY = "hobby"  # 爱好活动
    SELF_CARE = "self_care"  # 自我护理（护肤、化妆等）
    OTHER = "other"  # 其他


@dataclass
class SceneVariation:
    """
    场景变体 - 同一时间段内的不同瞬间

    用于间隔补充触发时提供变化，保持同一时间段内多次发送不重复。
    场景变体保持相同的地点和服装，但改变姿势、动作、表情等。

    Attributes:
        variation_id: 变体唯一标识
        description: 变体描述（中文，如"喝水休息"）
        pose: 姿势描述（英文）
        body_action: 身体动作（英文）
        hand_action: 手部动作（英文）
        expression: 表情（英文）
        mood: 情绪
        caption_theme: 配文主题（中文）
        is_used: 是否已使用
        used_at: 使用时间
    """

    variation_id: str
    description: str
    pose: str
    body_action: str
    hand_action: str
    expression: str
    mood: str = "neutral"
    caption_theme: str = ""
    is_used: bool = False
    used_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "variation_id": self.variation_id,
            "description": self.description,
            "pose": self.pose,
            "body_action": self.body_action,
            "hand_action": self.hand_action,
            "expression": self.expression,
            "mood": self.mood,
            "caption_theme": self.caption_theme,
            "is_used": self.is_used,
            "used_at": self.used_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SceneVariation":
        """从字典创建实例"""
        return cls(
            variation_id=data.get("variation_id", ""),
            description=data.get("description", ""),
            pose=data.get("pose", ""),
            body_action=data.get("body_action", ""),
            hand_action=data.get("hand_action", ""),
            expression=data.get("expression", ""),
            mood=data.get("mood", "neutral"),
            caption_theme=data.get("caption_theme", ""),
            is_used=data.get("is_used", False),
            used_at=data.get("used_at"),
        )

    def mark_used(self) -> None:
        """标记为已使用"""
        self.is_used = True
        self.used_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class ScheduleEntry:
    """
    日程条目 - 描述某个时间点的完整场景

    这是新架构的核心数据结构，包含生成自拍所需的所有维度信息。

    Attributes:
        time_point: 触发时间点，格式 "HH:MM"
        time_range_start: 时间范围开始，格式 "HH:MM"
        time_range_end: 时间范围结束，格式 "HH:MM"
        activity_type: 活动类型
        activity_description: 活动描述（中文）
        activity_detail: 详细说明
        location: 地点名称（中文）
        location_prompt: SD提示词用的地点描述（英文）
        pose: 姿势描述（英文）
        body_action: 身体动作（英文）
        hand_action: 手部动作（英文）
        expression: 表情（英文）
        mood: 情绪状态
        outfit: 服装（英文）
        accessories: 配饰（英文）
        environment: 环境描述（英文）
        lighting: 光线描述（英文）
        weather_context: 天气相关上下文（英文）
        caption_type: 配文类型
        suggested_caption_theme: 配文主题建议（中文）
        is_completed: 是否已完成
        completed_at: 完成时间
    """

    # 时间信息
    time_point: str  # 触发时间点 "HH:MM"
    time_range_start: str  # 时间范围开始
    time_range_end: str  # 时间范围结束

    # 活动信息
    activity_type: ActivityType
    activity_description: str  # 活动描述
    activity_detail: str  # 详细说明

    # 地点信息
    location: str  # 地点名称
    location_prompt: str  # SD提示词用的地点描述

    # 姿势与动作
    pose: str  # 姿势描述
    body_action: str  # 身体动作
    hand_action: str  # 手部动作

    # 表情与情绪
    expression: str  # 表情
    mood: str  # 情绪状态

    # 外观
    outfit: str  # 服装
    accessories: str  # 配饰

    # 环境
    environment: str  # 环境描述
    lighting: str  # 光线
    weather_context: str  # 天气相关上下文

    # 配文
    caption_type: str  # 配文类型
    suggested_caption_theme: str  # 配文主题建议

    # 状态
    is_completed: bool = False
    completed_at: Optional[str] = None

    # v2.0 新增：场景变体列表
    scene_variations: List[SceneVariation] = field(default_factory=list)
    # 追踪间隔补充使用情况
    interval_use_count: int = 0  # 间隔补充使用次数
    last_interval_use_at: Optional[str] = None  # 最后一次间隔补充使用时间

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典，用于 JSON 序列化

        Returns:
            包含所有属性的字典
        """
        return {
            "time_point": self.time_point,
            "time_range_start": self.time_range_start,
            "time_range_end": self.time_range_end,
            "activity_type": self.activity_type.value
            if isinstance(self.activity_type, ActivityType)
            else self.activity_type,
            "activity_description": self.activity_description,
            "activity_detail": self.activity_detail,
            "location": self.location,
            "location_prompt": self.location_prompt,
            "pose": self.pose,
            "body_action": self.body_action,
            "hand_action": self.hand_action,
            "expression": self.expression,
            "mood": self.mood,
            "outfit": self.outfit,
            "accessories": self.accessories,
            "environment": self.environment,
            "lighting": self.lighting,
            "weather_context": self.weather_context,
            "caption_type": self.caption_type,
            "suggested_caption_theme": self.suggested_caption_theme,
            "is_completed": self.is_completed,
            "completed_at": self.completed_at,
            # v2.0 新增字段
            "scene_variations": [v.to_dict() for v in self.scene_variations],
            "interval_use_count": self.interval_use_count,
            "last_interval_use_at": self.last_interval_use_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScheduleEntry":
        """
        从字典创建实例

        Args:
            data: 包含条目数据的字典

        Returns:
            ScheduleEntry 实例
        """
        # 处理 activity_type，支持字符串和枚举
        activity_type_value = data.get("activity_type", "other")
        if isinstance(activity_type_value, str):
            try:
                activity_type = ActivityType(activity_type_value)
            except ValueError:
                logger.warning(f"未知的活动类型: {activity_type_value}，使用 OTHER")
                activity_type = ActivityType.OTHER
        else:
            activity_type = activity_type_value

        # 解析场景变体列表
        variations_data = data.get("scene_variations", [])
        scene_variations = [SceneVariation.from_dict(v) for v in variations_data] if variations_data else []

        return cls(
            time_point=data.get("time_point", ""),
            time_range_start=data.get("time_range_start", ""),
            time_range_end=data.get("time_range_end", ""),
            activity_type=activity_type,
            activity_description=data.get("activity_description", ""),
            activity_detail=data.get("activity_detail", ""),
            location=data.get("location", ""),
            location_prompt=data.get("location_prompt", ""),
            pose=data.get("pose", ""),
            body_action=data.get("body_action", ""),
            hand_action=data.get("hand_action", ""),
            expression=data.get("expression", ""),
            mood=data.get("mood", "neutral"),
            outfit=data.get("outfit", ""),
            accessories=data.get("accessories", ""),
            environment=data.get("environment", ""),
            lighting=data.get("lighting", ""),
            weather_context=data.get("weather_context", ""),
            caption_type=data.get("caption_type", "share"),
            suggested_caption_theme=data.get("suggested_caption_theme", ""),
            is_completed=data.get("is_completed", False),
            completed_at=data.get("completed_at"),
            # v2.0 新增字段
            scene_variations=scene_variations,
            interval_use_count=data.get("interval_use_count", 0),
            last_interval_use_at=data.get("last_interval_use_at"),
        )

    def to_image_prompt(self) -> str:
        """
        将场景转换为 Stable Diffusion 图片生成提示词

        Returns:
            完整的英文提示词
        """
        prompt_parts = [
            # 强制主体
            "(1girl:1.4), (solo:1.3)",
            # 表情
            f"({self.expression}:1.2)" if self.expression else "",
            # 姿势与动作
            self.pose,
            self.body_action,
            f"({self.hand_action}:1.3)" if self.hand_action else "",
            # 服装
            self.outfit,
            self.accessories,
            # 环境
            self.location_prompt,
            self.environment,
            self.lighting,
            # 自拍视角
            "front camera view, looking at camera, selfie POV",
        ]

        # 过滤空值并拼接
        prompt_parts = [p for p in prompt_parts if p and p.strip()]
        return ", ".join(prompt_parts)

    def is_time_in_range(self, current_time: datetime) -> bool:
        """
        检查指定时间是否在本条目的时间范围内

        Args:
            current_time: 要检查的时间

        Returns:
            是否在范围内
        """
        current_str = current_time.strftime("%H:%M")
        return self._is_time_in_range_static(current_str, self.time_range_start, self.time_range_end)

    @staticmethod
    def _is_time_in_range_static(current: str, start: str, end: str) -> bool:
        """
        静态方法：检查时间是否在范围内，支持跨午夜

        Args:
            current: 当前时间 "HH:MM"
            start: 范围开始 "HH:MM"
            end: 范围结束 "HH:MM"

        Returns:
            是否在范围内
        """

        def time_to_minutes(time_str: str) -> int:
            try:
                parts = time_str.split(":")
                return int(parts[0]) * 60 + int(parts[1])
            except (ValueError, IndexError):
                return 0

        current_mins = time_to_minutes(current)
        start_mins = time_to_minutes(start)
        end_mins = time_to_minutes(end)

        if end_mins < start_mins:
            # 跨午夜
            return current_mins >= start_mins or current_mins <= end_mins
        else:
            return start_mins <= current_mins <= end_mins

    # ================================================================
    # v2.0 新增：场景变体相关方法
    # ================================================================

    def get_unused_variation(self) -> Optional[SceneVariation]:
        """
        获取一个未使用的场景变体

        Returns:
            未使用的场景变体，如果都已使用则返回 None
        """
        for variation in self.scene_variations:
            if not variation.is_used:
                return variation
        return None

    def get_next_variation(self) -> Optional[SceneVariation]:
        """
        获取下一个可用的场景变体（优先返回未使用的，否则重置并返回第一个）

        Returns:
            下一个可用的场景变体
        """
        # 首先尝试获取未使用的
        unused = self.get_unused_variation()
        if unused:
            return unused

        # 如果都已使用，重置所有变体并返回第一个
        if self.scene_variations:
            self.reset_variations()
            return self.scene_variations[0]

        return None

    def mark_variation_used(self, variation_id: str) -> bool:
        """
        标记指定变体为已使用

        Args:
            variation_id: 变体ID

        Returns:
            是否成功标记
        """
        for variation in self.scene_variations:
            if variation.variation_id == variation_id:
                variation.mark_used()
                return True
        return False

    def reset_variations(self) -> None:
        """重置所有变体的使用状态"""
        for variation in self.scene_variations:
            variation.is_used = False
            variation.used_at = None

    def get_used_variation_count(self) -> int:
        """获取已使用的变体数量"""
        return sum(1 for v in self.scene_variations if v.is_used)

    def has_available_variation(self) -> bool:
        """检查是否有可用的变体"""
        return any(not v.is_used for v in self.scene_variations)

    def record_interval_use(self) -> None:
        """记录一次间隔补充使用"""
        self.interval_use_count += 1
        self.last_interval_use_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def create_variation_prompt(self, variation: SceneVariation) -> str:
        """
        使用变体信息创建图片生成提示词

        保持原条目的地点、服装、环境等，替换姿势、动作、表情。

        Args:
            variation: 场景变体

        Returns:
            完整的英文提示词
        """
        prompt_parts = [
            # 强制主体
            "(1girl:1.4), (solo:1.3)",
            # 表情（使用变体的表情）
            f"({variation.expression}:1.2)" if variation.expression else "",
            # 姿势与动作（使用变体的姿势和动作）
            variation.pose,
            variation.body_action,
            f"({variation.hand_action}:1.3)" if variation.hand_action else "",
            # 服装（保持原条目的服装）
            self.outfit,
            self.accessories,
            # 环境（保持原条目的环境）
            self.location_prompt,
            self.environment,
            self.lighting,
            # 自拍视角
            "front camera view, looking at camera, selfie POV",
        ]

        # 过滤空值并拼接
        prompt_parts = [p for p in prompt_parts if p and p.strip()]
        return ", ".join(prompt_parts)


@dataclass
class DailyNarrativeState:
    """
    每日叙事状态 - 追踪一天的连续性

    用于保持一天内自拍的连续性，记录当前位置、服装、情绪变化等。
    在间隔补充触发时，根据叙事状态选择合适的场景。

    Attributes:
        current_location: 当前位置（最后一次发送时的位置）
        current_outfit: 当前服装（最后一次发送时的服装）
        mood_trajectory: 情绪变化轨迹
        sent_scenes_summary: 已发送场景的摘要列表
        last_sent_time: 上次发送时间
        last_sent_entry_time_point: 上次使用的日程条目时间点
        total_sent_count: 今日总发送次数
        interval_sent_count: 间隔补充发送次数
    """

    current_location: str = ""
    current_outfit: str = ""
    mood_trajectory: List[str] = field(default_factory=list)
    sent_scenes_summary: List[str] = field(default_factory=list)
    last_sent_time: str = ""
    last_sent_entry_time_point: str = ""
    total_sent_count: int = 0
    interval_sent_count: int = 0

    def update_after_send(
        self,
        entry: "ScheduleEntry",
        variation: Optional[SceneVariation] = None,
        is_interval: bool = False,
    ) -> None:
        """
        发送后更新叙事状态

        Args:
            entry: 使用的日程条目
            variation: 使用的场景变体（如果有）
            is_interval: 是否是间隔补充发送
        """
        self.current_location = entry.location
        self.current_outfit = entry.outfit
        self.last_sent_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.last_sent_entry_time_point = entry.time_point
        self.total_sent_count += 1

        if is_interval:
            self.interval_sent_count += 1

        # 记录情绪
        mood = variation.mood if variation else entry.mood
        self.mood_trajectory.append(mood)

        # 记录场景摘要
        if variation:
            summary = f"[{entry.time_point}] {variation.description}"
        else:
            summary = f"[{entry.time_point}] {entry.activity_description}"
        self.sent_scenes_summary.append(summary)

        # 限制摘要数量（保留最近 10 条）
        if len(self.sent_scenes_summary) > 10:
            self.sent_scenes_summary = self.sent_scenes_summary[-10:]
        if len(self.mood_trajectory) > 10:
            self.mood_trajectory = self.mood_trajectory[-10:]

    def get_context_for_caption(self) -> str:
        """
        获取用于生成配文的上下文

        Returns:
            叙事上下文字符串
        """
        if not self.sent_scenes_summary:
            return "今天还没有发过自拍。"

        context_parts = [f"今天已发送 {self.total_sent_count} 张自拍："]
        for summary in self.sent_scenes_summary[-3:]:  # 最近 3 条
            context_parts.append(f"- {summary}")

        if self.current_location:
            context_parts.append(f"当前位置：{self.current_location}")

        return "\n".join(context_parts)

    def can_transition_to(self, target_entry: "ScheduleEntry") -> Tuple[bool, str]:
        """
        检查是否可以自然过渡到目标场景

        Args:
            target_entry: 目标日程条目

        Returns:
            Tuple[是否可以过渡, 原因说明]
        """
        # 如果没有发送过，任何场景都可以
        if not self.last_sent_entry_time_point:
            return True, "首次发送"

        # 如果位置相同，可以过渡
        if self.current_location == target_entry.location:
            return True, "位置相同"

        # 如果位置不同，检查是否是合理的过渡
        # 例如：办公室 -> 家里 需要有"下班"的过渡
        # 这里简化处理，允许任何过渡但返回说明
        return True, f"位置变化：{self.current_location} -> {target_entry.location}"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "current_location": self.current_location,
            "current_outfit": self.current_outfit,
            "mood_trajectory": self.mood_trajectory,
            "sent_scenes_summary": self.sent_scenes_summary,
            "last_sent_time": self.last_sent_time,
            "last_sent_entry_time_point": self.last_sent_entry_time_point,
            "total_sent_count": self.total_sent_count,
            "interval_sent_count": self.interval_sent_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DailyNarrativeState":
        """从字典创建实例"""
        return cls(
            current_location=data.get("current_location", ""),
            current_outfit=data.get("current_outfit", ""),
            mood_trajectory=data.get("mood_trajectory", []),
            sent_scenes_summary=data.get("sent_scenes_summary", []),
            last_sent_time=data.get("last_sent_time", ""),
            last_sent_entry_time_point=data.get("last_sent_entry_time_point", ""),
            total_sent_count=data.get("total_sent_count", 0),
            interval_sent_count=data.get("interval_sent_count", 0),
        )


@dataclass
class DailySchedule:
    """
    每日日程 - 包含一天的所有场景条目

    由 LLM 在每天开始时生成，包含完整的日程安排。

    Attributes:
        date: 日期，格式 "YYYY-MM-DD"
        day_of_week: 星期几
        is_holiday: 是否假期
        weather: 天气
        character_persona: 角色设定
        entries: 日程条目列表
        generated_at: 生成时间
        model_used: 使用的模型
    """

    date: str  # 日期 YYYY-MM-DD
    day_of_week: str  # 星期几
    is_holiday: bool  # 是否假期
    weather: str  # 天气
    character_persona: str  # 角色设定
    entries: List[ScheduleEntry] = field(default_factory=list)
    generated_at: str = ""  # 生成时间
    model_used: str = ""  # 使用的模型

    # 当 model_used == "fallback" 时写入，说明触发回退的原因（用于验收与排查）
    fallback_reason: Optional[str] = None

    # 当 model_used == "fallback" 时写入：对应的失败包文件名（位于插件目录下）
    fallback_failure_package: Optional[str] = None

    # v2.0 新增：叙事状态追踪
    narrative_state: DailyNarrativeState = field(default_factory=DailyNarrativeState)

    def get_current_entry(self, current_time: Optional[datetime] = None) -> Optional[ScheduleEntry]:
        """
        获取当前时间应该触发的场景条目

        Args:
            current_time: 当前时间，None 则使用系统时间

        Returns:
            匹配的场景条目，如果没有匹配或已完成则返回 None
        """
        if current_time is None:
            current_time = datetime.now()

        current_time_str = current_time.strftime("%H:%M")

        for entry in self.entries:
            if entry.is_completed:
                continue

            if self._is_time_in_range(current_time_str, entry.time_range_start, entry.time_range_end):
                return entry

        return None

    def get_closest_entry(self, current_time: Optional[datetime] = None) -> tuple[Optional[ScheduleEntry], str]:
        """
        获取距离当前时间最近的场景条目（不考虑完成状态）

        用于间隔补充触发时，当没有精确匹配的条目时，找到时间上最接近的条目。
        返回条目和时间关系（before/after/within）。

        Args:
            current_time: 当前时间，None 则使用系统时间

        Returns:
            Tuple[条目, 时间关系]:
                - 条目: 最接近的 ScheduleEntry，没有则返回 None
                - 时间关系: "before"（当前时间在条目时间点之前）,
                           "after"（当前时间在条目时间点之后）,
                           "within"（当前时间在条目时间范围内）
        """
        if current_time is None:
            current_time = datetime.now()

        if not self.entries:
            return None, ""

        current_time_str = current_time.strftime("%H:%M")
        current_mins = self._time_to_minutes(current_time_str)

        closest_entry: Optional[ScheduleEntry] = None
        min_distance = float("inf")
        time_relation = ""

        for entry in self.entries:
            # 首先检查是否在条目的时间范围内
            if self._is_time_in_range(current_time_str, entry.time_range_start, entry.time_range_end):
                return entry, "within"

            # 计算与条目时间点的距离
            entry_mins = self._time_to_minutes(entry.time_point)

            # 计算距离（考虑跨午夜）
            distance = abs(current_mins - entry_mins)
            if distance > 720:  # 超过12小时，取另一个方向
                distance = 1440 - distance

            if distance < min_distance:
                min_distance = distance
                closest_entry = entry
                # 判断时间关系
                if current_mins < entry_mins:
                    # 当前时间在条目时间点之前
                    # 需要考虑跨午夜情况
                    if entry_mins - current_mins <= 720:
                        time_relation = "before"
                    else:
                        time_relation = "after"
                else:
                    # 当前时间在条目时间点之后
                    if current_mins - entry_mins <= 720:
                        time_relation = "after"
                    else:
                        time_relation = "before"

        return closest_entry, time_relation

    def get_next_entry(self, current_time: Optional[datetime] = None) -> Optional[ScheduleEntry]:
        """
        获取下一个未完成的场景条目

        Args:
            current_time: 当前时间，None 则使用系统时间

        Returns:
            下一个待执行的条目，没有则返回 None
        """
        if current_time is None:
            current_time = datetime.now()

        current_time_str = current_time.strftime("%H:%M")
        current_mins = self._time_to_minutes(current_time_str)

        for entry in self.entries:
            if entry.is_completed:
                continue

            entry_start_mins = self._time_to_minutes(entry.time_range_start)
            if entry_start_mins > current_mins:
                return entry

        return None

    def mark_entry_completed(self, time_point: str, caption: str = "") -> bool:
        """
        标记条目为已完成

        Args:
            time_point: 要标记的时间点
            caption: 实际生成的配文

        Returns:
            是否成功标记
        """
        for entry in self.entries:
            if entry.time_point == time_point:
                entry.is_completed = True
                entry.completed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                logger.info(f"日程条目已完成: {time_point}")
                return True

        logger.warning(f"未找到时间点为 {time_point} 的日程条目")
        return False

    def get_narrative_context(self, max_entries: int = 3) -> str:
        """
        获取叙事上下文（已完成的场景摘要）

        Args:
            max_entries: 最多返回多少条记录

        Returns:
            叙事上下文字符串
        """
        completed = [e for e in self.entries if e.is_completed][-max_entries:]

        if not completed:
            return "今天还没有发过自拍。"

        context_parts = [f"今天是{self.date}，{self.day_of_week}"]
        for entry in completed:
            context_parts.append(f"- [{entry.time_point}] {entry.activity_description}")

        return "\n".join(context_parts)

    def _is_time_in_range(self, current: str, start: str, end: str) -> bool:
        """
        检查时间是否在范围内，支持跨午夜

        Args:
            current: 当前时间 "HH:MM"
            start: 范围开始 "HH:MM"
            end: 范围结束 "HH:MM"

        Returns:
            是否在范围内
        """
        current_mins = self._time_to_minutes(current)
        start_mins = self._time_to_minutes(start)
        end_mins = self._time_to_minutes(end)

        if end_mins < start_mins:
            # 跨午夜
            return current_mins >= start_mins or current_mins <= end_mins
        else:
            return start_mins <= current_mins <= end_mins

    def _time_to_minutes(self, time_str: str) -> int:
        """
        时间字符串转分钟数

        Args:
            time_str: 时间字符串 "HH:MM"

        Returns:
            从 00:00 起的分钟数
        """
        try:
            parts = time_str.split(":")
            return int(parts[0]) * 60 + int(parts[1])
        except (ValueError, IndexError):
            logger.warning(f"无效的时间格式: {time_str}")
            return 0

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典，用于 JSON 序列化

        Returns:
            包含所有属性的字典
        """
        return {
            "date": self.date,
            "day_of_week": self.day_of_week,
            "is_holiday": self.is_holiday,
            "weather": self.weather,
            "character_persona": self.character_persona,
            "entries": [e.to_dict() for e in self.entries],
            "generated_at": self.generated_at,
            "model_used": self.model_used,
            # Phase 0：fallback 诊断信息
            "fallback_reason": self.fallback_reason,
            "fallback_failure_package": self.fallback_failure_package,
            # v2.0 新增
            "narrative_state": self.narrative_state.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DailySchedule":
        """
        从字典创建实例

        Args:
            data: 包含日程数据的字典

        Returns:
            DailySchedule 实例
        """
        # 解析叙事状态
        narrative_state_data = data.get("narrative_state", {})
        narrative_state = (
            DailyNarrativeState.from_dict(narrative_state_data) if narrative_state_data else DailyNarrativeState()
        )

        schedule = cls(
            date=data.get("date", ""),
            day_of_week=data.get("day_of_week", ""),
            is_holiday=data.get("is_holiday", False),
            weather=data.get("weather", ""),
            character_persona=data.get("character_persona", ""),
            generated_at=data.get("generated_at", ""),
            model_used=data.get("model_used", ""),
            fallback_reason=data.get("fallback_reason"),
            fallback_failure_package=data.get("fallback_failure_package"),
            narrative_state=narrative_state,
        )

        # 解析条目
        entries_data = data.get("entries", [])
        schedule.entries = [ScheduleEntry.from_dict(e) for e in entries_data]

        return schedule

    def save_to_file(self, filepath: str) -> bool:
        """
        保存日程到文件

        Args:
            filepath: 文件路径

        Returns:
            是否保存成功
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

            logger.info(f"日程已保存到: {filepath}")
            return True

        except Exception as e:
            logger.error(f"保存日程失败: {e}")
            return False

    @classmethod
    def load_from_file(cls, filepath: str) -> Optional["DailySchedule"]:
        """
        从文件加载日程

        Args:
            filepath: 文件路径

        Returns:
            DailySchedule 实例，如果加载失败返回 None
        """
        try:
            if not os.path.exists(filepath):
                logger.debug(f"日程文件不存在: {filepath}")
                return None

            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            schedule = cls.from_dict(data)
            logger.info(f"日程已从文件加载: {filepath}")
            return schedule

        except json.JSONDecodeError as e:
            logger.error(f"日程文件 JSON 解析失败: {e}")
            return None
        except Exception as e:
            logger.error(f"加载日程失败: {e}")
            return None

    def get_completed_count(self) -> int:
        """
        获取已完成条目数量

        Returns:
            已完成的条目数
        """
        return sum(1 for e in self.entries if e.is_completed)

    def get_pending_count(self) -> int:
        """
        获取待执行条目数量

        Returns:
            待执行的条目数
        """
        return sum(1 for e in self.entries if not e.is_completed)
