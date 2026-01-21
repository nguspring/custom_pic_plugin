"""
日程生成器模块

使用 LLM 动态生成每日日程，支持根据时间点生成完整的场景描述。
"""

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.common.logger import get_logger

from .schedule_models import ActivityType, DailySchedule, ScheduleEntry, SceneVariation

logger = get_logger("ScheduleGenerator")


# Prompt 模板 v2.0 - 支持场景变体
SCHEDULE_GENERATION_PROMPT = """今天是{date}，{day_of_week}，天气{weather}。
{holiday_note}

请为一个可爱的女孩规划今天的以下时间点的活动，每个时间点需要包含完整的场景描述：
时间点列表：{schedule_times}

对于每个时间点，请提供以下信息（JSON格式）：
{{
  "time_point": "HH:MM",
  "time_range_start": "HH:MM",
  "time_range_end": "HH:MM",
  "activity_type": "活动类型",
  "activity_description": "活动描述（中文）",
  "activity_detail": "详细说明这个时间点你在做什么（中文）",
  "location": "地点名称（中文）",
  "location_prompt": "英文地点描述，用于图片生成",
  "pose": "姿势描述（英文）",
  "body_action": "身体动作（英文）",
  "hand_action": "手部动作（英文）",
  "expression": "表情（英文）",
  "mood": "情绪",
  "outfit": "服装描述（英文）",
  "accessories": "配饰（英文）",
  "environment": "环境描述（英文）",
  "lighting": "光线描述（英文）",
  "weather_context": "天气相关描述（英文）",
  "caption_type": "NARRATIVE/ASK/SHARE/MONOLOGUE",
  "suggested_caption_theme": "配文主题建议（中文）",
  "scene_variations": [
    {{
      "variation_id": "v1",
      "description": "变体描述（中文，如'喝水休息'）",
      "pose": "姿势（英文）",
      "body_action": "身体动作（英文）",
      "hand_action": "手部动作（英文）",
      "expression": "表情（英文）",
      "mood": "情绪",
      "caption_theme": "配文主题（中文）"
    }}
  ]
}}

## 活动类型选项
- sleeping: 睡觉
- waking_up: 起床
- eating: 用餐
- working: 工作
- studying: 学习
- exercising: 运动
- relaxing: 休闲放松
- socializing: 社交
- commuting: 通勤
- hobby: 爱好活动
- self_care: 自我护理
- other: 其他

## 配文类型选项
- NARRATIVE: 叙事式（延续故事线）
- ASK: 询问式（征求意见）
- SHARE: 分享式（分享心情）
- MONOLOGUE: 独白式（自言自语）

## 场景变体说明（重要！）
每个时间点必须包含 2-3 个场景变体（scene_variations），用于在该时间段内的多次发送。
变体规则：
1. 变体保持相同的地点（location）和服装（outfit）
2. 变体改变姿势、动作、表情，提供不同的"瞬间"
3. 变体描述符合当前活动的自然行为
4. 变体之间应有明显区别，避免重复

变体示例（工作时间段）：
- v1: 认真敲键盘，专注工作
- v2: 伸懒腰，眼睛有点累
- v3: 喝水休息，看着屏幕发呆

变体示例（午餐时间段）：
- v1: 夹菜吃饭，满足的表情
- v2: 拿手机拍食物
- v3: 吃完收拾，擦嘴巴

## 重要规则
1. 活动安排符合真实生活逻辑，一天有连续性
2. 场景描述生动具体，适合生成自拍图片
3. 表情和情绪要与活动匹配
4. 时间范围应该合理（通常1-2小时）
5. 手部动作必须与当前活动场景匹配，不要使用通用手势
6. 所有英文提示词使用 Stable Diffusion 风格的 tag 格式
7. 每个条目必须包含 2-3 个不同的场景变体
8. 返回有效的JSON数组

## 禁止事项（非常重要！）
这是自拍场景，手机在画面外拍摄，因此：
1. 【禁止】在 body_action、hand_action 中使用 phone、smartphone、device、mobile 等词汇
2. 【禁止】描述"刷手机"、"看手机"、"拿手机"、"玩手机"等动作
3. 【禁止】使用 scrolling phone、holding phone、using phone 等表达
4. 【替代方案】如果想表达放松/发呆状态，使用：zoning out、staring blankly、daydreaming、resting eyes 等
5. 【替代方案】如果想表达休息状态，使用：stretching、yawning、resting head on hand、playing with hair 等

请返回完整的日程JSON数组（只返回JSON，不要包含其他文字）：
"""


class ScheduleGenerator:
    """
    日程生成器 - 使用 LLM 生成每日日程

    负责调用 LLM 生成每日日程，并验证输出格式。
    支持回退到静态模板作为备用方案。
    """

    PROMPT_VERSION = "1.0"

    def __init__(self, plugin_instance: Any):
        """
        初始化生成器

        Args:
            plugin_instance: 插件实例，用于读取配置和调用 LLM API
        """
        self.plugin = plugin_instance
        logger.info("ScheduleGenerator 初始化完成")

    def _get_schedule_persona_block(self) -> str:
        """获取日程人设配置并构建人设提示块
        
        根据用户配置的人设描述和生活习惯，构建注入到日程生成 prompt 中的人设块。
        
        Returns:
            构建好的人设提示块字符串，如果未启用则返回空字符串
        """
        # 检查是否启用日程人设注入
        persona_enabled = self.plugin.get_config("auto_selfie.schedule_persona_enabled", True)
        
        if not persona_enabled:
            logger.debug("日程人设注入未启用")
            return ""
        
        # 获取人设配置
        persona_text = self.plugin.get_config(
            "auto_selfie.schedule_persona_text",
            "是一个大二女大学生"
        )
        lifestyle = self.plugin.get_config(
            "auto_selfie.schedule_lifestyle",
            "作息规律，喜欢宅家但偶尔也会出门"
        )
        
        # 如果两个配置都为空，返回空字符串
        if not persona_text and not lifestyle:
            logger.debug("日程人设和生活习惯配置均为空，跳过注入")
            return ""
        
        # 构建人设块
        persona_block_parts = []
        
        if persona_text:
            persona_block_parts.append(f"她{persona_text}")
        
        if lifestyle:
            persona_block_parts.append(f"生活习惯：{lifestyle}")
        
        persona_block = "。".join(persona_block_parts)
        
        logger.info(f"日程人设注入已启用，人设: {persona_block[:50]}...")
        logger.debug(f"日程人设块内容: {persona_block}")
        
        return persona_block

    async def generate_daily_schedule(
        self,
        date: str,
        schedule_times: List[str],
        weather: str = "晴天",
        is_holiday: bool = False,
    ) -> Optional[DailySchedule]:
        """
        生成每日日程

        Args:
            date: 日期 YYYY-MM-DD
            schedule_times: 配置的时间点列表 ["08:00", "12:00", "20:00"]
            weather: 天气
            is_holiday: 是否假期

        Returns:
            DailySchedule 或 None（失败时）
        """
        logger.info(f"开始生成日程: {date}, 时间点数量: {len(schedule_times)}")

        try:
            # 计算星期几
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            weekday_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
            day_of_week = weekday_names[date_obj.weekday()]

            # 如果是周末且未明确设置，自动判断为假期
            if date_obj.weekday() >= 5 and not is_holiday:
                is_holiday = True
                logger.debug("周末自动设置为假期模式")

            # 构建 Prompt
            prompt = self._build_generation_prompt(
                schedule_times=schedule_times,
                day_of_week=day_of_week,
                weather=weather,
                is_holiday=is_holiday,
                date=date,
            )

            logger.debug(f"生成的 Prompt 长度: {len(prompt)}")

            # 调用 LLM
            response = await self._call_llm(prompt)

            if not response:
                logger.warning("LLM 返回空响应，使用回退方案")
                return self._generate_fallback_schedule(
                    date=date,
                    day_of_week=day_of_week,
                    is_holiday=is_holiday,
                    weather=weather,
                    schedule_times=schedule_times,
                )

            # 解析响应
            schedule = self._parse_llm_response(
                response=response,
                date=date,
                day_of_week=day_of_week,
                is_holiday=is_holiday,
                weather=weather,
            )

            if schedule and self._validate_schedule(schedule):
                logger.info(
                    f"日程生成成功，共 {len(schedule.entries)} 个条目"
                )
                return schedule
            else:
                logger.warning("日程解析或验证失败，使用回退方案")
                return self._generate_fallback_schedule(
                    date=date,
                    day_of_week=day_of_week,
                    is_holiday=is_holiday,
                    weather=weather,
                    schedule_times=schedule_times,
                )

        except Exception as e:
            logger.error(f"生成日程异常: {e}")
            import traceback

            logger.debug(f"异常堆栈: {traceback.format_exc()}")
            return self._generate_fallback_schedule(
                date=date,
                day_of_week=day_of_week if "day_of_week" in dir() else "未知",
                is_holiday=is_holiday,
                weather=weather,
                schedule_times=schedule_times,
            )

    def _build_generation_prompt(
        self,
        schedule_times: List[str],
        day_of_week: str,
        weather: str,
        is_holiday: bool,
        date: str,
    ) -> str:
        """
        构建生成日程的 Prompt

        Args:
            schedule_times: 时间点列表
            day_of_week: 星期几
            weather: 天气
            is_holiday: 是否假期
            date: 日期

        Returns:
            完整的 Prompt 字符串
        """
        holiday_note = (
            "今天是假期/周末，可以安排更轻松的活动。"
            if is_holiday
            else "今天是工作日。"
        )

        # 获取人设块
        persona_block = self._get_schedule_persona_block()
        
        # 构建基础 prompt
        prompt = SCHEDULE_GENERATION_PROMPT.format(
            date=date,
            day_of_week=day_of_week,
            weather=weather,
            holiday_note=holiday_note,
            schedule_times=", ".join(schedule_times),
        )
        
        # 如果有人设，在 prompt 中注入人设信息
        # 将 "请为一个可爱的女孩" 替换为包含人设的描述
        if persona_block:
            persona_insert = f"请为一个可爱的女孩规划今天的以下时间点的活动。{persona_block}。\n\n每个时间点需要包含完整的场景描述"
            prompt = prompt.replace(
                "请为一个可爱的女孩规划今天的以下时间点的活动，每个时间点需要包含完整的场景描述",
                persona_insert
            )
            logger.debug(f"日程生成 Prompt 已注入人设信息")

        return prompt

    async def _call_llm(self, prompt: str) -> Optional[str]:
        """
        调用 LLM 生成内容

        使用 MaiBot 的 llm_api 进行调用。
        优先使用 planner 模型（需要规划能力），其次是 replyer。

        Args:
            prompt: 完整的提示词

        Returns:
            生成的内容，失败时返回 None
        """
        from src.config.config import model_config as maibot_model_config
        from src.llm_models.utils_model import LLMRequest
        from src.plugin_system.apis import llm_api

        logger.debug(f"调用 LLM，prompt 长度: {len(prompt)}")

        try:
            # 获取用户配置的自定义模型 ID
            custom_model_id = self.plugin.get_config(
                "auto_selfie.schedule_model_id", ""
            )
            logger.debug(f"配置的自定义模型ID: '{custom_model_id}'")

            # 如果用户配置了自定义模型，尝试使用
            if custom_model_id:
                available_models = llm_api.get_available_models()
                if custom_model_id in available_models:
                    model_config = available_models[custom_model_id]
                    logger.info(f"使用用户配置的模型: {custom_model_id}")

                    success, content, reasoning, model_name = (
                        await llm_api.generate_with_model(
                            prompt=prompt,
                            model_config=model_config,
                            request_type="plugin.auto_selfie.schedule_generate",
                            temperature=0.7,
                            max_tokens=4000,
                        )
                    )

                    if success and content:
                        logger.debug(f"LLM 生成成功，使用模型: {model_name}")
                        return content
                    else:
                        logger.warning(f"LLM 生成失败: {content}")
                else:
                    logger.warning(
                        f"配置的模型 '{custom_model_id}' 不存在，"
                        "回退到默认模型"
                    )

            # 默认使用 MaiBot 的 planner 模型（规划模型）
            # 因为日程生成需要规划能力
            logger.debug("尝试使用 MaiBot planner 模型")
            try:
                planner_request = LLMRequest(
                    model_set=maibot_model_config.model_task_config.planner,
                    request_type="plugin.auto_selfie.schedule_generate",
                )

                content, reasoning = await planner_request.generate_response_async(
                    prompt,
                    temperature=0.7,
                    max_tokens=4000,
                )

                if content:
                    logger.debug("LLM 生成成功，使用 MaiBot planner 模型")
                    return content
                else:
                    logger.warning("planner 模型生成失败，返回空内容")

            except Exception as e:
                logger.warning(f"使用 planner 模型失败: {e}，尝试 replyer 模型")

                # 尝试使用 replyer 作为备用
                try:
                    replyer_request = LLMRequest(
                        model_set=maibot_model_config.model_task_config.replyer,
                        request_type="plugin.auto_selfie.schedule_generate",
                    )

                    content, reasoning = (
                        await replyer_request.generate_response_async(
                            prompt,
                            temperature=0.7,
                            max_tokens=4000,
                        )
                    )

                    if content:
                        logger.debug("LLM 生成成功，使用 MaiBot replyer 模型")
                        return content

                except Exception as e2:
                    logger.warning(f"使用 replyer 模型也失败: {e2}")

                # 最后尝试使用 llm_api 的第一个可用模型
                available_models = llm_api.get_available_models()
                if available_models:
                    first_key = next(iter(available_models))
                    model_config = available_models[first_key]
                    logger.debug(f"使用备用模型: {first_key}")

                    success, content, reasoning, model_name = (
                        await llm_api.generate_with_model(
                            prompt=prompt,
                            model_config=model_config,
                            request_type="plugin.auto_selfie.schedule_generate",
                            temperature=0.7,
                            max_tokens=4000,
                        )
                    )

                    if success and content:
                        return content

            return None

        except Exception as e:
            logger.error(f"LLM 调用异常: {e}")
            import traceback

            logger.debug(f"异常堆栈: {traceback.format_exc()}")
            return None

    def _parse_llm_response(
        self,
        response: str,
        date: str,
        day_of_week: str,
        is_holiday: bool,
        weather: str,
    ) -> Optional[DailySchedule]:
        """
        解析 LLM 响应为 DailySchedule

        Args:
            response: LLM 响应字符串
            date: 日期
            day_of_week: 星期几
            is_holiday: 是否假期
            weather: 天气

        Returns:
            DailySchedule 实例，失败时返回 None
        """
        logger.debug(f"开始解析 LLM 响应，长度: {len(response)}")

        try:
            # 尝试提取 JSON 数组
            json_content = self._extract_json_array(response)

            if not json_content:
                logger.warning("未能从响应中提取 JSON 数组")
                return None

            entries_data = json.loads(json_content)

            if not isinstance(entries_data, list):
                logger.warning("解析结果不是数组")
                return None

            # 创建日程
            schedule = DailySchedule(
                date=date,
                day_of_week=day_of_week,
                is_holiday=is_holiday,
                weather=weather,
                character_persona="",  # 不再使用角色人设
                generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                model_used="llm",
            )

            # 解析每个条目
            for i, entry_data in enumerate(entries_data):
                try:
                    entry = self._parse_entry(entry_data)
                    if entry:
                        schedule.entries.append(entry)
                        logger.debug(f"成功解析条目 {i}: {entry.time_point}")
                    else:
                        logger.warning(f"条目 {i} 解析失败，跳过")
                except Exception as e:
                    logger.warning(f"解析条目 {i} 异常: {e}")
                    continue

            if not schedule.entries:
                logger.warning("没有成功解析任何条目")
                return None

            return schedule

        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析失败: {e}")
            return None
        except Exception as e:
            logger.error(f"解析响应异常: {e}")
            return None

    def _extract_json_array(self, text: str) -> Optional[str]:
        """
        从文本中提取 JSON 数组

        Args:
            text: 包含 JSON 的文本

        Returns:
            提取的 JSON 字符串，失败时返回 None
        """
        # 尝试直接匹配数组
        array_match = re.search(r"\[\s*\{[\s\S]*?\}\s*\]", text)
        if array_match:
            return array_match.group()

        # 尝试匹配 markdown 代码块中的 JSON
        code_block_match = re.search(
            r"```(?:json)?\s*(\[\s*\{[\s\S]*?\}\s*\])\s*```", text
        )
        if code_block_match:
            return code_block_match.group(1)

        # 尝试查找任何看起来像 JSON 数组的内容
        # 寻找第一个 [ 和最后一个 ]
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            potential_json = text[start : end + 1]
            try:
                json.loads(potential_json)
                return potential_json
            except json.JSONDecodeError:
                pass

        return None

    def _parse_entry(self, data: Dict[str, Any]) -> Optional[ScheduleEntry]:
        """
        解析单个日程条目

        Args:
            data: 条目数据字典

        Returns:
            ScheduleEntry 实例，失败时返回 None
        """
        required_fields = [
            "time_point",
            "activity_type",
            "activity_description",
        ]

        for field in required_fields:
            if field not in data or not data[field]:
                logger.warning(f"条目缺少必要字段: {field}")
                return None

        # 设置默认时间范围
        time_point = data.get("time_point", "")
        if "time_range_start" not in data or not data["time_range_start"]:
            data["time_range_start"] = self._adjust_time(time_point, -30)
        if "time_range_end" not in data or not data["time_range_end"]:
            data["time_range_end"] = self._adjust_time(time_point, 60)

        # 标准化 caption_type
        caption_type = data.get("caption_type", "SHARE")
        if isinstance(caption_type, str):
            caption_type = caption_type.upper()
            valid_types = ["NARRATIVE", "ASK", "SHARE", "MONOLOGUE", "NONE"]
            if caption_type not in valid_types:
                caption_type = "SHARE"
        data["caption_type"] = caption_type.lower()

        return ScheduleEntry.from_dict(data)

    def _adjust_time(self, time_str: str, minutes: int) -> str:
        """
        调整时间

        Args:
            time_str: 原始时间 "HH:MM"
            minutes: 调整分钟数（正数往后，负数往前）

        Returns:
            调整后的时间字符串
        """
        try:
            parts = time_str.split(":")
            total_mins = int(parts[0]) * 60 + int(parts[1]) + minutes
            total_mins = max(0, min(1439, total_mins))  # 限制在 00:00-23:59
            return f"{total_mins // 60:02d}:{total_mins % 60:02d}"
        except (ValueError, IndexError):
            return time_str

    def _validate_schedule(self, schedule: DailySchedule) -> bool:
        """
        验证日程的有效性

        Args:
            schedule: 要验证的日程

        Returns:
            是否有效
        """
        if not schedule:
            return False

        if not schedule.entries:
            logger.warning("日程没有任何条目")
            return False

        # 检查时间点格式
        time_pattern = re.compile(r"^([01]?[0-9]|2[0-3]):[0-5][0-9]$")
        for entry in schedule.entries:
            if not time_pattern.match(entry.time_point):
                logger.warning(f"无效的时间点格式: {entry.time_point}")
                return False

        # 检查是否有重复时间点
        time_points = [e.time_point for e in schedule.entries]
        if len(time_points) != len(set(time_points)):
            logger.warning("存在重复的时间点")
            # 不返回 False，允许重复但记录警告

        logger.debug(f"日程验证通过，共 {len(schedule.entries)} 个条目")
        return True

    def _generate_fallback_schedule(
        self,
        date: str,
        day_of_week: str,
        is_holiday: bool,
        weather: str,
        schedule_times: List[str],
    ) -> DailySchedule:
        """
        生成回退日程（当 LLM 调用失败时使用）

        使用预定义的模板生成基础日程，包含场景变体。

        Args:
            date: 日期
            day_of_week: 星期几
            is_holiday: 是否假期
            weather: 天气
            schedule_times: 时间点列表

        Returns:
            DailySchedule 实例
        """
        logger.info("使用回退方案生成日程")

        schedule = DailySchedule(
            date=date,
            day_of_week=day_of_week,
            is_holiday=is_holiday,
            weather=weather,
            character_persona="",  # 不再使用角色人设
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_used="fallback",
        )

        # 预定义的场景模板（带变体）
        fallback_scenes = self._get_fallback_scenes(is_holiday)

        # 为每个时间点分配场景
        for i, time_point in enumerate(schedule_times):
            scene_index = i % len(fallback_scenes)
            scene = fallback_scenes[scene_index]

            # 解析场景变体
            scene_variations = []
            if "scene_variations" in scene:
                for var_data in scene["scene_variations"]:
                    variation = SceneVariation(
                        variation_id=var_data.get("variation_id", f"v{len(scene_variations)+1}"),
                        description=var_data.get("description", ""),
                        pose=var_data.get("pose", ""),
                        body_action=var_data.get("body_action", ""),
                        hand_action=var_data.get("hand_action", ""),
                        expression=var_data.get("expression", ""),
                        mood=var_data.get("mood", ""),
                        caption_theme=var_data.get("caption_theme", ""),
                    )
                    scene_variations.append(variation)

            entry = ScheduleEntry(
                time_point=time_point,
                time_range_start=self._adjust_time(time_point, -30),
                time_range_end=self._adjust_time(time_point, 60),
                activity_type=scene["activity_type"],
                activity_description=scene["activity_description"],
                activity_detail=scene.get("activity_detail", ""),
                location=scene["location"],
                location_prompt=scene["location_prompt"],
                pose=scene["pose"],
                body_action=scene["body_action"],
                hand_action=scene["hand_action"],
                expression=scene["expression"],
                mood=scene["mood"],
                outfit=scene["outfit"],
                accessories=scene.get("accessories", ""),
                environment=scene["environment"],
                lighting=scene["lighting"],
                weather_context=scene.get("weather_context", ""),
                caption_type=scene["caption_type"],
                suggested_caption_theme=scene["suggested_caption_theme"],
                scene_variations=scene_variations,
            )
            schedule.entries.append(entry)

        logger.info(f"回退日程生成完成，共 {len(schedule.entries)} 个条目（每条目含变体）")
        return schedule

    def _get_fallback_scenes(self, is_holiday: bool) -> List[Dict[str, Any]]:
        """
        获取回退场景模板（带场景变体）

        Args:
            is_holiday: 是否假期

        Returns:
            场景模板列表，每个场景包含 2-3 个变体
        """
        if is_holiday:
            # 假期/周末场景
            return [
                {
                    "activity_type": ActivityType.WAKING_UP,
                    "activity_description": "周末懒觉醒来",
                    "activity_detail": "今天睡到自然醒，真舒服",
                    "location": "卧室",
                    "location_prompt": "bedroom, cozy room, morning light",
                    "pose": "sitting on bed, stretching",
                    "body_action": "just woke up, relaxed",
                    "hand_action": "rubbing eyes, sleepy",
                    "expression": "sleepy smile, messy hair",
                    "mood": "relaxed",
                    "outfit": "pajamas, oversized shirt",
                    "accessories": "messy bed hair",
                    "environment": "cozy bedroom, soft bedding",
                    "lighting": "soft morning light",
                    "weather_context": "sunny morning",
                    "caption_type": "narrative",
                    "suggested_caption_theme": "分享周末懒觉的惬意",
                    "scene_variations": [
                        {
                            "variation_id": "v1",
                            "description": "伸懒腰打哈欠",
                            "pose": "arms stretched up, yawning",
                            "body_action": "stretching whole body",
                            "hand_action": "arms raised above head",
                            "expression": "yawning, eyes half closed",
                            "mood": "sleepy",
                            "caption_theme": "刚醒来的困意",
                        },
                        {
                            "variation_id": "v2",
                            "description": "看窗外阳光",
                            "pose": "sitting on bed, looking at window",
                            "body_action": "leaning against pillow",
                            "hand_action": "hands resting on lap",
                            "expression": "surprised, wide eyes",
                            "mood": "surprised",
                            "caption_theme": "发现已经睡到中午了",
                        },
                    ],
                },
                {
                    "activity_type": ActivityType.EATING,
                    "activity_description": "周末早午餐",
                    "activity_detail": "给自己做了一顿丰盛的早午餐",
                    "location": "家里餐厅",
                    "location_prompt": "dining room, home, brunch",
                    "pose": "sitting at table",
                    "body_action": "eating, enjoying food",
                    "hand_action": "holding fork, eating",
                    "expression": "happy smile, satisfied",
                    "mood": "happy",
                    "outfit": "casual home clothes",
                    "accessories": "",
                    "environment": "warm home atmosphere",
                    "lighting": "natural daylight",
                    "weather_context": "",
                    "caption_type": "share",
                    "suggested_caption_theme": "分享美味的早午餐",
                    "scene_variations": [
                        {
                            "variation_id": "v1",
                            "description": "吃得很满足",
                            "pose": "sitting at table, leaning back",
                            "body_action": "chewing food",
                            "hand_action": "holding fork with food",
                            "expression": "happy eating face, cheeks puffed",
                            "mood": "happy",
                            "caption_theme": "好吃！",
                        },
                        {
                            "variation_id": "v2",
                            "description": "欣赏食物",
                            "pose": "leaning forward, admiring food",
                            "body_action": "looking at the delicious meal",
                            "hand_action": "hands clasped under chin",
                            "expression": "focused, slight smile",
                            "mood": "excited",
                            "caption_theme": "看起来好好吃",
                        },
                        {
                            "variation_id": "v3",
                            "description": "喝饮料休息",
                            "pose": "sitting back, relaxed",
                            "body_action": "taking a break from eating",
                            "hand_action": "holding cup with both hands",
                            "expression": "content, peaceful",
                            "mood": "peaceful",
                            "caption_theme": "吃饱喝足真幸福",
                        },
                    ],
                },
                {
                    "activity_type": ActivityType.RELAXING,
                    "activity_description": "下午休闲时光",
                    "activity_detail": "窝在沙发上看剧",
                    "location": "客厅",
                    "location_prompt": "living room, couch, cozy",
                    "pose": "lounging on sofa",
                    "body_action": "relaxing, watching TV",
                    "hand_action": "holding remote control",
                    "expression": "relaxed, content",
                    "mood": "peaceful",
                    "outfit": "comfortable clothes",
                    "accessories": "",
                    "environment": "cozy living room",
                    "lighting": "soft indoor light",
                    "weather_context": "",
                    "caption_type": "monologue",
                    "suggested_caption_theme": "周末放松时刻",
                    "scene_variations": [
                        {
                            "variation_id": "v1",
                            "description": "认真看剧",
                            "pose": "curled up on sofa",
                            "body_action": "focused on screen",
                            "hand_action": "holding remote, pressing",
                            "expression": "concentrated, slight frown",
                            "mood": "focused",
                            "caption_theme": "追剧中勿扰",
                        },
                        {
                            "variation_id": "v2",
                            "description": "躺着发呆",
                            "pose": "lying sideways on sofa",
                            "body_action": "staring blankly, daydreaming",
                            "hand_action": "hand resting under cheek",
                            "expression": "blank stare, relaxed",
                            "mood": "lazy",
                            "caption_theme": "无聊中...",
                        },
                        {
                            "variation_id": "v3",
                            "description": "吃零食",
                            "pose": "sitting cross-legged on sofa",
                            "body_action": "snacking while watching",
                            "hand_action": "reaching into snack bag",
                            "expression": "happy munching",
                            "mood": "happy",
                            "caption_theme": "看剧必须配零食",
                        },
                    ],
                },
                {
                    "activity_type": ActivityType.SELF_CARE,
                    "activity_description": "晚间护肤",
                    "activity_detail": "认真做晚间护肤",
                    "location": "浴室",
                    "location_prompt": "bathroom, mirror, skincare",
                    "pose": "standing at mirror",
                    "body_action": "doing skincare routine",
                    "hand_action": "applying skincare product",
                    "expression": "focused, peaceful",
                    "mood": "peaceful",
                    "outfit": "bathrobe",
                    "accessories": "hair band",
                    "environment": "clean bathroom",
                    "lighting": "warm bathroom light",
                    "weather_context": "",
                    "caption_type": "share",
                    "suggested_caption_theme": "护肤日常",
                    "scene_variations": [
                        {
                            "variation_id": "v1",
                            "description": "涂面膜",
                            "pose": "leaning close to mirror",
                            "body_action": "applying face mask",
                            "hand_action": "spreading mask on face",
                            "expression": "concentrated, lips pursed",
                            "mood": "focused",
                            "caption_theme": "面膜时间",
                        },
                        {
                            "variation_id": "v2",
                            "description": "拍打精华",
                            "pose": "standing, head tilted",
                            "body_action": "patting face gently",
                            "hand_action": "both hands patting cheeks",
                            "expression": "eyes closed, relaxed",
                            "mood": "peaceful",
                            "caption_theme": "让精华好好吸收",
                        },
                    ],
                },
            ]
        else:
            # 工作日场景
            return [
                {
                    "activity_type": ActivityType.WAKING_UP,
                    "activity_description": "早起准备",
                    "activity_detail": "闹钟响了，新的一天开始",
                    "location": "卧室",
                    "location_prompt": "bedroom, morning, waking up",
                    "pose": "sitting on bed edge",
                    "body_action": "just woke up",
                    "hand_action": "rubbing eyes",
                    "expression": "sleepy, yawning",
                    "mood": "sleepy",
                    "outfit": "pajamas",
                    "accessories": "messy hair",
                    "environment": "bedroom, morning light",
                    "lighting": "soft morning light",
                    "weather_context": "morning",
                    "caption_type": "narrative",
                    "suggested_caption_theme": "早安问候",
                    "scene_variations": [
                        {
                            "variation_id": "v1",
                            "description": "关闹钟",
                            "pose": "reaching for phone on nightstand",
                            "body_action": "still lying in bed",
                            "hand_action": "tapping phone to turn off alarm",
                            "expression": "annoyed, sleepy",
                            "mood": "grumpy",
                            "caption_theme": "又是被闹钟叫醒的一天",
                        },
                        {
                            "variation_id": "v2",
                            "description": "坐起来发呆",
                            "pose": "sitting on bed, hunched",
                            "body_action": "trying to wake up",
                            "hand_action": "hands on knees",
                            "expression": "blank stare, half asleep",
                            "mood": "groggy",
                            "caption_theme": "需要咖啡...",
                        },
                    ],
                },
                {
                    "activity_type": ActivityType.EATING,
                    "activity_description": "午餐时间",
                    "activity_detail": "中午休息吃个饭",
                    "location": "餐厅",
                    "location_prompt": "restaurant, lunch, eating",
                    "pose": "sitting at table",
                    "body_action": "having lunch",
                    "hand_action": "holding chopsticks",
                    "expression": "happy, enjoying",
                    "mood": "happy",
                    "outfit": "casual work clothes",
                    "accessories": "",
                    "environment": "restaurant, lunch time",
                    "lighting": "natural daylight",
                    "weather_context": "",
                    "caption_type": "share",
                    "suggested_caption_theme": "午餐分享",
                    "scene_variations": [
                        {
                            "variation_id": "v1",
                            "description": "认真吃饭",
                            "pose": "sitting, focused on food",
                            "body_action": "eating steadily",
                            "hand_action": "chopsticks picking up food",
                            "expression": "enjoying, mouth slightly open",
                            "mood": "satisfied",
                            "caption_theme": "今天的午餐不错",
                        },
                        {
                            "variation_id": "v2",
                            "description": "和同事聊天",
                            "pose": "sitting, turned slightly",
                            "body_action": "talking while eating",
                            "hand_action": "chopsticks held, gesturing",
                            "expression": "laughing, animated",
                            "mood": "cheerful",
                            "caption_theme": "午餐摸鱼时间",
                        },
                        {
                            "variation_id": "v3",
                            "description": "喝汤休息",
                            "pose": "sitting, holding bowl",
                            "body_action": "taking a break from eating",
                            "hand_action": "both hands holding soup bowl",
                            "expression": "content, eyes closed, savoring",
                            "mood": "peaceful",
                            "caption_theme": "喝口汤暖暖的",
                        },
                    ],
                },
                {
                    "activity_type": ActivityType.WORKING,
                    "activity_description": "下午工作",
                    "activity_detail": "认真工作中",
                    "location": "办公室",
                    "location_prompt": "office, desk, working",
                    "pose": "sitting at desk",
                    "body_action": "working, typing",
                    "hand_action": "typing on keyboard",
                    "expression": "focused, professional",
                    "mood": "focused",
                    "outfit": "work clothes",
                    "accessories": "",
                    "environment": "modern office",
                    "lighting": "office lighting",
                    "weather_context": "",
                    "caption_type": "ask",
                    "suggested_caption_theme": "工作日常",
                    "scene_variations": [
                        {
                            "variation_id": "v1",
                            "description": "敲键盘工作",
                            "pose": "leaning forward, focused",
                            "body_action": "typing intensely",
                            "hand_action": "fingers on keyboard, typing fast",
                            "expression": "concentrated, slight frown",
                            "mood": "focused",
                            "caption_theme": "认真工作中",
                        },
                        {
                            "variation_id": "v2",
                            "description": "伸懒腰休息",
                            "pose": "leaning back in chair, stretching",
                            "body_action": "taking a break",
                            "hand_action": "arms stretched overhead",
                            "expression": "tired, eyes closed",
                            "mood": "tired",
                            "caption_theme": "工作累了休息一下",
                        },
                        {
                            "variation_id": "v3",
                            "description": "喝咖啡提神",
                            "pose": "sitting back, relaxed",
                            "body_action": "taking a coffee break",
                            "hand_action": "holding coffee cup",
                            "expression": "contemplative, looking at screen",
                            "mood": "thoughtful",
                            "caption_theme": "咖啡续命",
                        },
                    ],
                },
                {
                    "activity_type": ActivityType.RELAXING,
                    "activity_description": "下班回家",
                    "activity_detail": "终于下班啦",
                    "location": "家里",
                    "location_prompt": "home, evening, relaxing",
                    "pose": "relaxed pose",
                    "body_action": "resting, relaxing",
                    "hand_action": "hands resting on lap",
                    "expression": "tired but happy",
                    "mood": "relaxed",
                    "outfit": "casual clothes",
                    "accessories": "",
                    "environment": "cozy home",
                    "lighting": "warm evening light",
                    "weather_context": "evening",
                    "caption_type": "monologue",
                    "suggested_caption_theme": "下班后的放松",
                    "scene_variations": [
                        {
                            "variation_id": "v1",
                            "description": "瘫在沙发上",
                            "pose": "sprawled on sofa",
                            "body_action": "completely relaxed",
                            "hand_action": "arms spread out",
                            "expression": "exhausted but relieved",
                            "mood": "exhausted",
                            "caption_theme": "终于到家了",
                        },
                        {
                            "variation_id": "v2",
                            "description": "换家居服",
                            "pose": "standing, changing clothes",
                            "body_action": "putting on comfortable clothes",
                            "hand_action": "pulling on hoodie",
                            "expression": "relieved, comfortable",
                            "mood": "comfortable",
                            "caption_theme": "换上舒服的衣服",
                        },
                        {
                            "variation_id": "v3",
                            "description": "躺着放空发呆",
                            "pose": "lying on sofa, legs up",
                            "body_action": "zoning out, staring at ceiling",
                            "hand_action": "hands resting on stomach",
                            "expression": "blank, zoned out",
                            "mood": "lazy",
                            "caption_theme": "下班后的摸鱼时间",
                        },
                    ],
                },
            ]

    def get_schedule_file_path(self, date: Optional[str] = None) -> str:
        """
        获取日程文件路径

        Args:
            date: 日期，None 则使用今天

        Returns:
            日程文件的完整路径
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        # 获取插件目录
        plugin_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(plugin_dir, f"daily_schedule_{date}.json")

    def _cleanup_old_schedule_files(self, current_date: str) -> None:
        """
        清理旧的日程文件
        
        删除非当天的日程文件，避免文件越堆越多。
        
        Args:
            current_date: 当前日期 YYYY-MM-DD
        """
        try:
            # 获取插件目录
            plugin_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            # 查找所有日程文件
            import glob
            pattern = os.path.join(plugin_dir, "daily_schedule_*.json")
            schedule_files = glob.glob(pattern)
            
            deleted_count = 0
            for file_path in schedule_files:
                # 提取文件中的日期
                filename = os.path.basename(file_path)
                # daily_schedule_YYYY-MM-DD.json
                if filename.startswith("daily_schedule_") and filename.endswith(".json"):
                    file_date = filename[15:-5]  # 提取日期部分
                    
                    # 如果不是当天的文件，删除它
                    if file_date != current_date:
                        try:
                            os.remove(file_path)
                            deleted_count += 1
                            logger.debug(f"已删除旧日程文件: {filename}")
                        except OSError as e:
                            logger.warning(f"删除旧日程文件失败 {filename}: {e}")
            
            if deleted_count > 0:
                logger.info(f"已清理 {deleted_count} 个旧日程文件")
                
        except Exception as e:
            logger.warning(f"清理旧日程文件时出错: {e}")

    async def get_or_generate_schedule(
        self,
        date: str,
        schedule_times: List[str],
        weather: str = "晴天",
        is_holiday: bool = False,
        force_regenerate: bool = False,
    ) -> Optional[DailySchedule]:
        """
        获取或生成日程

        首先尝试从文件加载，如果不存在或需要强制重新生成，则调用 LLM 生成。
        同时会清理非当天的旧日程文件。

        Args:
            date: 日期
            schedule_times: 时间点列表
            weather: 天气
            is_holiday: 是否假期
            force_regenerate: 是否强制重新生成

        Returns:
            DailySchedule 实例
        """
        # 清理旧的日程文件（非当天的）
        self._cleanup_old_schedule_files(date)
        
        file_path = self.get_schedule_file_path(date)

        # 如果不是强制重新生成，尝试从文件加载
        if not force_regenerate:
            existing = DailySchedule.load_from_file(file_path)
            if existing and existing.date == date:
                logger.info(f"从文件加载已有日程: {date}")
                return existing

        # 生成新日程（角色信息在 generate_daily_schedule 内部自动获取）
        schedule = await self.generate_daily_schedule(
            date=date,
            schedule_times=schedule_times,
            weather=weather,
            is_holiday=is_holiday,
        )

        # 保存到文件
        if schedule:
            schedule.save_to_file(file_path)

        return schedule
