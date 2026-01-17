"""
动态日程系统数据模型模块

包含活动类型枚举、日程条目和每日日程等数据类型，
用于 LLM 动态生成日程的新架构。
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

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
                logger.warning(
                    f"未知的活动类型: {activity_type_value}，使用 OTHER"
                )
                activity_type = ActivityType.OTHER
        else:
            activity_type = activity_type_value

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

    def get_current_entry(
        self, current_time: Optional[datetime] = None
    ) -> Optional[ScheduleEntry]:
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

            if self._is_time_in_range(
                current_time_str, entry.time_range_start, entry.time_range_end
            ):
                return entry

        return None

    def get_next_entry(
        self, current_time: Optional[datetime] = None
    ) -> Optional[ScheduleEntry]:
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

    def mark_entry_completed(
        self, time_point: str, caption: str = ""
    ) -> bool:
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
            context_parts.append(
                f"- [{entry.time_point}] {entry.activity_description}"
            )

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
        schedule = cls(
            date=data.get("date", ""),
            day_of_week=data.get("day_of_week", ""),
            is_holiday=data.get("is_holiday", False),
            weather=data.get("weather", ""),
            character_persona=data.get("character_persona", ""),
            generated_at=data.get("generated_at", ""),
            model_used=data.get("model_used", ""),
        )

        # 解析条目
        entries_data = data.get("entries", [])
        schedule.entries = [
            ScheduleEntry.from_dict(e) for e in entries_data
        ]

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
