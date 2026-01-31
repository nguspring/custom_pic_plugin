"""
日程生成器模块

使用 LLM 动态生成每日日程，支持根据时间点生成完整的场景描述。
"""

import json
import os
import re
import traceback
import uuid
import copy
import hashlib
from datetime import datetime, timedelta
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
- v2: 闻一闻香味，期待地夹菜
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
        """初始化生成器

        Args:
            plugin_instance: 插件实例，用于读取配置和调用 LLM API
        """
        self.plugin = plugin_instance
        # Phase 0：用于 fallback 失败包记录（模型选择路径/最后一次调用信息）
        self._last_llm_debug: Dict[str, Any] = {}
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
        persona_text = self.plugin.get_config("auto_selfie.schedule_persona_text", "是一个大二女大学生")
        lifestyle = self.plugin.get_config("auto_selfie.schedule_lifestyle", "作息规律，喜欢宅家但偶尔也会出门")

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

    def _get_schedule_persona_signature(self) -> str:
        """生成当前日程人设的签名。

        用于缓存失效：当人设/生活习惯/开关变化时，能够触发重新生成日程。

        Returns:
            可序列化、稳定的签名字符串
        """
        persona_enabled = self.plugin.get_config("auto_selfie.schedule_persona_enabled", True)
        persona_text = self.plugin.get_config("auto_selfie.schedule_persona_text", "")
        lifestyle = self.plugin.get_config("auto_selfie.schedule_lifestyle", "")

        signature_obj = {
            "prompt_version": self.PROMPT_VERSION,
            "persona_enabled": bool(persona_enabled),
            "persona_text": str(persona_text or ""),
            "lifestyle": str(lifestyle or ""),
        }
        return json.dumps(signature_obj, ensure_ascii=False, sort_keys=True)

    def _get_schedule_persona_constraints_block(self) -> str:
        """根据人设文本补充约束，避免生成出戏日程（例如学生人设却像上班族）。"""
        persona_enabled = self.plugin.get_config("auto_selfie.schedule_persona_enabled", True)
        if not persona_enabled:
            return ""

        persona_text = str(self.plugin.get_config("auto_selfie.schedule_persona_text", "") or "")
        lifestyle = str(self.plugin.get_config("auto_selfie.schedule_lifestyle", "") or "")
        persona_text_l = persona_text.lower()

        is_student = any(
            k in persona_text_l for k in ["学生", "大学", "大一", "大二", "大三", "大四", "研究生", "高中", "初中"]
        )  # 中英混合也能匹配
        is_worker = any(k in persona_text_l for k in ["上班", "公司", "白领", "打工", "社畜", "职场", "同事", "办公室"])

        if is_student and not is_worker:
            return (
                "\n【身份约束】她是学生/在校生，工作日主要是上课、自习、社团/运动与生活琐事。\n"
                "- 不要安排‘上班/到公司/开会/写日报/同事聚餐’等职场情境。\n"
                "- 地点优先：教室、图书馆、宿舍、食堂、校园、社团活动室。\n"
                "- activity_type 优先使用 studying / commuting(去学校) / hobby / exercising / relaxing。\n"
            )

        if is_worker:
            return "\n【身份约束】她是上班族/职场人士，工作日可安排通勤、办公室工作、会议与下班后生活。\n"

        if lifestyle:
            return f"\n【生活习惯提示】{lifestyle}\n"

        return ""

    def _load_recent_schedule_context(self, *, date: str, days: int = 7) -> str:
        """加载最近 N 天的日程摘要，注入到生成 prompt，降低跨天重复。

        Returns:
            可直接拼接到 prompt 的中文上下文块（可能为空字符串）
        """
        try:
            base_date = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            return ""

        plugin_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        summaries: List[str] = []

        for i in range(1, days + 1):
            d = (base_date - timedelta(days=i)).strftime("%Y-%m-%d")
            file_path = os.path.join(plugin_dir, f"daily_schedule_{d}.json")
            schedule = DailySchedule.load_from_file(file_path)
            if not schedule or not schedule.entries:
                continue

            # 摘要只保留“时间点 + 活动描述 + 地点”，避免泄露过多提示词
            items: List[str] = []
            for e in schedule.entries[:12]:
                items.append(f"[{e.time_point}] {e.activity_description} @ {e.location}")
            summaries.append(f"{d}: " + "；".join(items))

        if not summaries:
            return ""

        return (
            "\n【过去7天回顾（用于去重）】\n"
            "下面是最近几天的日程摘要，请你生成今天的日程时尽量避免高度相似的活动组合/重复场景：\n"
            + "\n".join(summaries)
            + "\n"
        )

    def _save_schedule_fallback_failure_package(
        self,
        *,
        date: str,
        fallback_reason: str,
        prompt: str,
        response: Optional[str],
        exception_stack: str,
    ) -> str:
        """保存日程 fallback 失败包。

        失败包内容用于验收与排查：prompt/response/异常堆栈/模型选择路径。

        Returns:
            失败包文件名（相对于插件根目录的路径）
        """
        plugin_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        package_dir = os.path.join(plugin_dir, "fallback_packages", "schedule")
        os.makedirs(package_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_id = uuid.uuid4().hex[:8]
        filename = f"schedule_fallback_{date}_{timestamp}_{short_id}.json"
        file_path = os.path.join(package_dir, filename)

        payload: Dict[str, Any] = {
            "type": "daily_schedule_fallback",
            "date": date,
            "fallback_reason": fallback_reason,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "prompt": prompt,
            "response": response,
            "exception_stack": exception_stack,
            "model_selection_path": self._last_llm_debug.get("selection_path", []),
            "debug": {
                "schedule_generator_model": self.plugin.get_config("auto_selfie.schedule_generator_model", ""),
                "schedule_model_id_legacy": self.plugin.get_config("auto_selfie.schedule_model_id", ""),
                "schedule_persona_enabled": self.plugin.get_config("auto_selfie.schedule_persona_enabled", True),
                "schedule_persona_text": self.plugin.get_config("auto_selfie.schedule_persona_text", ""),
                "schedule_lifestyle": self.plugin.get_config("auto_selfie.schedule_lifestyle", ""),
            },
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        # 返回相对路径，方便写回日程文件
        return os.path.join("fallback_packages", "schedule", filename)

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

        # Phase 0：用于失败包记录
        prompt: str = ""
        response: Optional[str] = None
        day_of_week: str = "未知"

        try:
            # 计算星期几
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            weekday_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
            day_of_week = weekday_names[date_obj.weekday()]

            # 如果是周末且未明确设置，自动判断为假期
            if date_obj.weekday() >= 5 and not is_holiday:
                is_holiday = True
                logger.debug("周末自动设置为假期模式")

            persona_signature = self._get_schedule_persona_signature()

            # 构建 Prompt
            prompt = self._build_generation_prompt(
                schedule_times=schedule_times,
                day_of_week=day_of_week,
                weather=weather,
                is_holiday=is_holiday,
                date=date,
            )

            # Phase 2：跨天去重（保留最近7天摘要回灌到 prompt）
            prompt = prompt + self._load_recent_schedule_context(date=date, days=7)

            logger.debug(f"生成的 Prompt 长度: {len(prompt)}")

            # 调用 LLM
            response = await self._call_llm(prompt)

            if not response:
                logger.warning("LLM 返回空响应，使用回退方案")
                fallback_reason = "llm_empty_response"
                failure_package = self._save_schedule_fallback_failure_package(
                    date=date,
                    fallback_reason=fallback_reason,
                    prompt=prompt,
                    response=response,
                    exception_stack="",
                )
                return self._generate_fallback_schedule(
                    date=date,
                    day_of_week=day_of_week,
                    is_holiday=is_holiday,
                    weather=weather,
                    schedule_times=schedule_times,
                    fallback_reason=fallback_reason,
                    failure_package=failure_package,
                    persona_signature=persona_signature,
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
                # 写入人设签名，支持缓存失效
                schedule.character_persona = persona_signature
                logger.info(f"日程生成成功，共 {len(schedule.entries)} 个条目")
                return schedule

            # 解析或验证失败 -> fallback
            fallback_reason = "parse_failed" if schedule is None else "validation_failed"
            logger.warning(f"日程解析或验证失败({fallback_reason})，使用回退方案")
            failure_package = self._save_schedule_fallback_failure_package(
                date=date,
                fallback_reason=fallback_reason,
                prompt=prompt,
                response=response,
                exception_stack="",
            )
            return self._generate_fallback_schedule(
                date=date,
                day_of_week=day_of_week,
                is_holiday=is_holiday,
                weather=weather,
                schedule_times=schedule_times,
                fallback_reason=fallback_reason,
                failure_package=failure_package,
                persona_signature=persona_signature,
            )

        except Exception as e:
            logger.error(f"生成日程异常: {e}")
            exception_stack = traceback.format_exc()
            logger.debug(f"异常堆栈: {exception_stack}")

            fallback_reason = "exception"
            failure_package = self._save_schedule_fallback_failure_package(
                date=date,
                fallback_reason=fallback_reason,
                prompt=prompt,
                response=response,
                exception_stack=exception_stack,
            )
            return self._generate_fallback_schedule(
                date=date,
                day_of_week=day_of_week,
                is_holiday=is_holiday,
                weather=weather,
                schedule_times=schedule_times,
                fallback_reason=fallback_reason,
                failure_package=failure_package,
                persona_signature=persona_signature,
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
        holiday_note = "今天是假期/周末，可以安排更轻松的活动。" if is_holiday else "今天是工作日。"

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
            constraints_block = self._get_schedule_persona_constraints_block()
            persona_insert = (
                f"请为一个可爱的女孩规划今天的以下时间点的活动。{persona_block}。\n"
                f"{constraints_block}\n"
                "每个时间点需要包含完整的场景描述"
            )
            prompt = prompt.replace(
                "请为一个可爱的女孩规划今天的以下时间点的活动，每个时间点需要包含完整的场景描述",
                persona_insert,
            )
            logger.debug("日程生成 Prompt 已注入人设信息")

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

        selection_path: List[Dict[str, Any]] = []

        def record(step: str, **kwargs: Any) -> None:
            selection_path.append({"step": step, **kwargs})

        try:
            # 获取用户配置的自定义模型 ID
            # 优先使用 config_schema 中的 schedule_generator_model（旧键名 schedule_model_id 仅作为兼容）
            custom_model_id = str(self.plugin.get_config("auto_selfie.schedule_generator_model", "") or "").strip()
            if not custom_model_id:
                custom_model_id = str(self.plugin.get_config("auto_selfie.schedule_model_id", "") or "").strip()

            logger.debug(f"配置的自定义模型ID: '{custom_model_id}'")
            record("config.custom_model_id", custom_model_id=custom_model_id)

            # 如果用户配置了自定义模型，尝试使用
            if custom_model_id:
                available_models = llm_api.get_available_models()
                record("llm_api.available_models", count=len(available_models), keys=list(available_models.keys()))
                if custom_model_id in available_models:
                    model_config = available_models[custom_model_id]
                    logger.info(f"使用用户配置的模型: {custom_model_id}")
                    record("llm_api.use_custom_model", model_id=custom_model_id)

                    success, content, reasoning, model_name = await llm_api.generate_with_model(
                        prompt=prompt,
                        model_config=model_config,
                        request_type="plugin.auto_selfie.schedule_generate",
                        temperature=0.7,
                        max_tokens=4000,
                    )
                    record(
                        "llm_api.custom_model.result",
                        success=success,
                        model_name=model_name,
                        content_len=len(content) if content else 0,
                    )

                    if success and content:
                        logger.debug(f"LLM 生成成功，使用模型: {model_name}")
                        self._last_llm_debug = {"selection_path": selection_path}
                        return content

                    logger.warning(f"LLM 生成失败: {content}")
                else:
                    logger.warning(f"配置的模型 '{custom_model_id}' 不存在，回退到默认模型")
                    record("llm_api.custom_model_not_found", model_id=custom_model_id)

            # 默认使用 MaiBot 的 planner 模型（规划模型）
            logger.debug("尝试使用 MaiBot planner 模型")
            record("maibot.planner.try")
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

                record("maibot.planner.result", content_len=len(content) if content else 0)
                if content:
                    logger.debug("LLM 生成成功，使用 MaiBot planner 模型")
                    self._last_llm_debug = {"selection_path": selection_path}
                    return content

                logger.warning("planner 模型生成失败，返回空内容")

            except Exception as e:
                logger.warning(f"使用 planner 模型失败: {e}，尝试 replyer 模型")
                record(
                    "maibot.planner.exception",
                    error=str(e),
                    stack=traceback.format_exc(),
                )

            # 尝试使用 replyer 作为备用
            record("maibot.replyer.try")
            try:
                replyer_request = LLMRequest(
                    model_set=maibot_model_config.model_task_config.replyer,
                    request_type="plugin.auto_selfie.schedule_generate",
                )

                content, reasoning = await replyer_request.generate_response_async(
                    prompt,
                    temperature=0.7,
                    max_tokens=4000,
                )

                record("maibot.replyer.result", content_len=len(content) if content else 0)
                if content:
                    logger.debug("LLM 生成成功，使用 MaiBot replyer 模型")
                    self._last_llm_debug = {"selection_path": selection_path}
                    return content

            except Exception as e2:
                logger.warning(f"使用 replyer 模型也失败: {e2}")
                record(
                    "maibot.replyer.exception",
                    error=str(e2),
                    stack=traceback.format_exc(),
                )

            # 最后尝试使用 llm_api 的第一个可用模型
            available_models = llm_api.get_available_models()
            record("llm_api.fallback_available_models", count=len(available_models), keys=list(available_models.keys()))
            if available_models:
                first_key = next(iter(available_models))
                model_config = available_models[first_key]
                logger.debug(f"使用备用模型: {first_key}")
                record("llm_api.use_first_available", model_id=first_key)

                success, content, reasoning, model_name = await llm_api.generate_with_model(
                    prompt=prompt,
                    model_config=model_config,
                    request_type="plugin.auto_selfie.schedule_generate",
                    temperature=0.7,
                    max_tokens=4000,
                )
                record(
                    "llm_api.first_available.result",
                    success=success,
                    model_name=model_name,
                    content_len=len(content) if content else 0,
                )

                if success and content:
                    self._last_llm_debug = {"selection_path": selection_path}
                    return content

            self._last_llm_debug = {"selection_path": selection_path}
            return None

        except Exception as e:
            logger.error(f"LLM 调用异常: {e}")
            record("_call_llm.exception", error=str(e), stack=traceback.format_exc())
            self._last_llm_debug = {"selection_path": selection_path}
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
        """解析 LLM 响应为 DailySchedule。

        注意：LLM 输出可能包含 markdown 代码块、前后缀说明文字，并且条目里可能包含嵌套数组
        （例如 scene_variations）。因此不能用简单正则截取最短 "[...]" 来提取 JSON。
        """
        logger.debug(f"开始解析 LLM 响应，长度: {len(response)}")

        json_content: Optional[str] = None

        try:
            # 尝试提取 JSON 数组
            json_content = self._extract_json_array(response)

            if not json_content:
                head = (response or "")[:240].replace("\n", "\\n")
                logger.warning(f"未能从响应中提取 JSON 数组，response_head={head}")
                return None

            logger.debug(
                "提取到 JSON 数组，长度=%s，head=%s",
                len(json_content),
                json_content[:160].replace("\n", "\\n"),
            )

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
            extracted_len = len(json_content) if json_content else 0
            logger.error(f"JSON 解析失败: {e} (extracted_len={extracted_len})")

            # 追加上下文，方便定位是“截断”还是“模型输出脏数据”
            if json_content and getattr(e, "pos", None) is not None:
                pos = int(e.pos)
                left = max(0, pos - 140)
                right = min(len(json_content), pos + 140)
                ctx = json_content[left:right].replace("\n", "\\n")
                logger.error(f"JSON 解析失败上下文(pos={pos}, range={left}:{right}): {ctx}")

            return None
        except Exception as e:
            logger.error(f"解析响应异常: {e}")
            return None

    def _extract_json_array(self, text: str) -> Optional[str]:
        """从文本中提取 JSON 数组。

        旧实现使用正则 `\\[\\s*\\{[\\s\\S]*?\\}\\s*\\]` 做“最短匹配”，
        在条目中存在嵌套数组（例如 scene_variations）时，会在内部 `]` 处提前截断，
        导致 json.loads() 失败，从而触发 parse_failed -> fallback。

        新实现优先使用 JSONDecoder.raw_decode 扫描，天然支持嵌套结构，且能忽略 JSON 之后的尾随文本。
        """
        if not text:
            return None

        def looks_like_schedule_entries(obj: Any) -> bool:
            if not isinstance(obj, list) or not obj:
                return False
            # 只看前几个元素即可，避免误把 scene_variations 之类的列表当成 entries
            for item in obj[:5]:
                if not isinstance(item, dict):
                    continue
                if item.get("time_point") and item.get("activity_type"):
                    return True
            return False

        # Strategy 1: 扫描所有 '['，尝试从该位置 raw_decode 出一个 JSON list
        decoder = json.JSONDecoder()
        for m in re.finditer(r"\[", text):
            idx = m.start()
            try:
                obj, end = decoder.raw_decode(text[idx:])
            except Exception:
                continue

            if looks_like_schedule_entries(obj):
                extracted = text[idx : idx + end]
                logger.debug(f"从响应中提取到 JSON 数组(raw_decode_scan): start={idx}, len={len(extracted)}")
                return extracted

        # Strategy 2: 兼容：如果文本里有完整 JSON 数组但不满足 looks_like（例如字段缺失），
        # 仍尝试从第一个 '[' 位置 raw_decode 出 list 作为兜底。
        first = text.find("[")
        if first != -1:
            try:
                obj, end = decoder.raw_decode(text[first:])
                if isinstance(obj, list):
                    extracted = text[first : first + end]
                    logger.debug(f"从响应中提取到 JSON 数组(raw_decode_first): start={first}, len={len(extracted)}")
                    return extracted
            except Exception:
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
        # 使用小窗口（±5分钟）确保在时间点附近准时触发
        time_point = data.get("time_point", "")
        if "time_range_start" not in data or not data["time_range_start"]:
            data["time_range_start"] = self._adjust_time(time_point, -5)
        if "time_range_end" not in data or not data["time_range_end"]:
            data["time_range_end"] = self._adjust_time(time_point, 5)

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
        fallback_reason: Optional[str] = None,
        failure_package: Optional[str] = None,
        persona_signature: str = "",
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
            character_persona=persona_signature,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_used="fallback",
            fallback_reason=fallback_reason,
            fallback_failure_package=failure_package,
        )

        # 预定义的场景模板（带变体）
        fallback_scenes = self._get_fallback_scenes(is_holiday=is_holiday, date=date)

        # 兜底修正：回退模板也必须遵守禁用规则（禁止 phone 等）
        fallback_scenes = self._sanitize_fallback_scenes(fallback_scenes)

        # 为每个时间点分配场景
        for i, time_point in enumerate(schedule_times):
            scene_index = i % len(fallback_scenes)
            scene = fallback_scenes[scene_index]

            # 解析场景变体
            scene_variations: List[SceneVariation] = []
            if "scene_variations" in scene:
                for var_data in scene["scene_variations"]:
                    if not isinstance(var_data, dict):
                        continue

                    nv = dict(var_data)
                    if not nv.get("variation_id"):
                        nv["variation_id"] = f"v{len(scene_variations) + 1}"

                    scene_variations.append(SceneVariation.from_dict(nv))

            entry = ScheduleEntry(
                time_point=time_point,
                time_range_start=self._adjust_time(time_point, -5),
                time_range_end=self._adjust_time(time_point, 5),
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

    def _sanitize_fallback_scenes(self, scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """清理回退模板中的禁用词，避免出现 phone/smartphone 等。

        说明：LLM prompt 已禁止，但回退模板是硬编码，必须自检。
        """
        banned_patterns = [
            re.compile(r"\bsmartphone\b", re.IGNORECASE),
            re.compile(r"\bphone\b", re.IGNORECASE),
            re.compile(r"\bmobile\b", re.IGNORECASE),
            re.compile(r"\bdevice\b", re.IGNORECASE),
        ]

        def sanitize_text(text: str) -> str:
            cleaned = text
            for pat in banned_patterns:
                cleaned = pat.sub("", cleaned)
            cleaned = re.sub(r",\s*,+", ", ", cleaned)
            cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,")
            return cleaned

        def sanitize_scene(scene: Dict[str, Any]) -> Dict[str, Any]:
            new_scene = dict(scene)
            for key in ["pose", "body_action", "hand_action", "location_prompt", "environment", "activity_detail"]:
                if key in new_scene and isinstance(new_scene[key], str):
                    new_scene[key] = sanitize_text(new_scene[key])

            variations = new_scene.get("scene_variations")
            if isinstance(variations, list):
                sanitized_vars: List[Dict[str, Any]] = []
                for v in variations:
                    if not isinstance(v, dict):
                        continue
                    nv = dict(v)
                    for key in ["pose", "body_action", "hand_action", "description"]:
                        if key in nv and isinstance(nv[key], str):
                            nv[key] = sanitize_text(nv[key])
                    sanitized_vars.append(nv)
                new_scene["scene_variations"] = sanitized_vars

            return new_scene

        return [sanitize_scene(s) for s in scenes if isinstance(s, dict)]

    def _get_fallback_scenes(self, *, is_holiday: bool, date: str) -> List[Dict[str, Any]]:
        """获取回退场景模板（带场景变体，Phase 2：多套模板）。

        共9个模板，适配默认的9个时间点：
        07:30, 09:00, 10:30, 12:00, 14:00, 16:00, 18:00, 20:00, 22:00

        Phase 2：fallback 模板“多套”实现方式：
        - 先构建一套 base 模板（原有硬编码）
        - 再基于 base 生成一套 variant（学生/上班族/假期户外等差异化）
        - 使用 date + persona 作为 key 做确定性选择（避免同一天多次生成时随机跳变）

        Args:
            is_holiday: 是否假期
            date: 日期 YYYY-MM-DD

        Returns:
            场景模板列表，每个场景包含 2-3 个变体
        """
        persona_text = str(self.plugin.get_config("auto_selfie.schedule_persona_text", "") or "")
        persona_text_l = persona_text.lower()
        is_student = any(
            k in persona_text_l for k in ["学生", "大学", "大一", "大二", "大三", "大四", "研究生", "高中", "初中"]
        )
        is_worker = any(k in persona_text_l for k in ["上班", "公司", "白领", "打工", "社畜", "职场", "同事", "办公室"])
        persona_mode = "student" if (is_student and not is_worker) else ("worker" if is_worker else "generic")

        base: List[Dict[str, Any]]

        if is_holiday:
            # 假期/周末场景 - 共9个，对应9个时间点
            base = [
                # ========== 1. 07:30 - 懒觉醒来 ==========
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
                # ========== 2. 09:00 - 早午餐 ==========
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
                    ],
                },
                # ========== 3. 10:30 - 上午休闲 ==========
                {
                    "activity_type": ActivityType.HOBBY,
                    "activity_description": "上午悠闲时光",
                    "activity_detail": "窝在沙发上看书听音乐",
                    "location": "客厅",
                    "location_prompt": "living room, sofa, reading",
                    "pose": "curled up on sofa with book",
                    "body_action": "reading leisurely",
                    "hand_action": "holding book",
                    "expression": "peaceful, focused",
                    "mood": "peaceful",
                    "outfit": "comfortable home clothes",
                    "accessories": "headphones",
                    "environment": "cozy living room, sunlight",
                    "lighting": "warm morning sunlight",
                    "weather_context": "sunny morning",
                    "caption_type": "share",
                    "suggested_caption_theme": "周末的悠闲时光",
                    "scene_variations": [
                        {
                            "variation_id": "v1",
                            "description": "专注看书",
                            "pose": "leaning against sofa cushion",
                            "body_action": "absorbed in reading",
                            "hand_action": "turning pages",
                            "expression": "concentrated, slight smile",
                            "mood": "focused",
                            "caption_theme": "这本书真好看",
                        },
                        {
                            "variation_id": "v2",
                            "description": "听音乐放空",
                            "pose": "head tilted, eyes closed",
                            "body_action": "enjoying music",
                            "hand_action": "adjusting headphones",
                            "expression": "peaceful, smiling",
                            "mood": "relaxed",
                            "caption_theme": "好听的歌让人心情好",
                        },
                    ],
                },
                # ========== 4. 12:00 - 午餐 ==========
                {
                    "activity_type": ActivityType.EATING,
                    "activity_description": "午餐时间",
                    "activity_detail": "点了外卖或自己做饭",
                    "location": "家里餐厅",
                    "location_prompt": "dining room, home, lunch",
                    "pose": "sitting at table",
                    "body_action": "having lunch",
                    "hand_action": "holding chopsticks",
                    "expression": "happy, hungry",
                    "mood": "happy",
                    "outfit": "casual home clothes",
                    "accessories": "",
                    "environment": "warm home atmosphere",
                    "lighting": "natural daylight",
                    "weather_context": "",
                    "caption_type": "share",
                    "suggested_caption_theme": "午餐吃什么",
                    "scene_variations": [
                        {
                            "variation_id": "v1",
                            "description": "开吃",
                            "pose": "leaning forward, eating",
                            "body_action": "picking up food",
                            "hand_action": "chopsticks picking up food",
                            "expression": "anticipating, mouth slightly open",
                            "mood": "excited",
                            "caption_theme": "开饭啦",
                        },
                        {
                            "variation_id": "v2",
                            "description": "吃饱了",
                            "pose": "leaning back, satisfied",
                            "body_action": "finished eating",
                            "hand_action": "hands resting on table",
                            "expression": "satisfied, content",
                            "mood": "content",
                            "caption_theme": "好饱~",
                        },
                    ],
                },
                # ========== 5. 14:00 - 下午看剧 ==========
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
                # ========== 6. 16:00 - 下午茶 ==========
                {
                    "activity_type": ActivityType.HOBBY,
                    "activity_description": "下午茶时光",
                    "activity_detail": "悠闲地喝下午茶",
                    "location": "阳台",
                    "location_prompt": "balcony, afternoon tea, plants",
                    "pose": "sitting in chair, relaxed",
                    "body_action": "enjoying tea time",
                    "hand_action": "holding tea cup",
                    "expression": "peaceful, content",
                    "mood": "peaceful",
                    "outfit": "casual dress",
                    "accessories": "",
                    "environment": "sunny balcony with plants",
                    "lighting": "warm afternoon sunlight",
                    "weather_context": "sunny afternoon",
                    "caption_type": "share",
                    "suggested_caption_theme": "惬意的下午茶",
                    "scene_variations": [
                        {
                            "variation_id": "v1",
                            "description": "品茶",
                            "pose": "leaning back, eyes closed",
                            "body_action": "savoring the tea",
                            "hand_action": "tea cup near lips",
                            "expression": "peaceful, satisfied",
                            "mood": "relaxed",
                            "caption_theme": "这杯茶真香",
                        },
                        {
                            "variation_id": "v2",
                            "description": "看风景发呆",
                            "pose": "chin resting on hand",
                            "body_action": "gazing into distance",
                            "hand_action": "elbow on table",
                            "expression": "dreamy, thoughtful",
                            "mood": "contemplative",
                            "caption_theme": "发呆也是一种享受",
                        },
                    ],
                },
                # ========== 7. 18:00 - 晚餐 ==========
                {
                    "activity_type": ActivityType.EATING,
                    "activity_description": "晚餐时间",
                    "activity_detail": "吃一顿丰盛的晚餐",
                    "location": "餐厅",
                    "location_prompt": "dining room, dinner, warm lighting",
                    "pose": "sitting at table",
                    "body_action": "having dinner",
                    "hand_action": "holding chopsticks",
                    "expression": "happy, satisfied",
                    "mood": "happy",
                    "outfit": "casual home clothes",
                    "accessories": "",
                    "environment": "warm dining atmosphere",
                    "lighting": "warm indoor light",
                    "weather_context": "evening",
                    "caption_type": "share",
                    "suggested_caption_theme": "今天的晚餐",
                    "scene_variations": [
                        {
                            "variation_id": "v1",
                            "description": "大口吃饭",
                            "pose": "leaning forward, eating",
                            "body_action": "enjoying the meal",
                            "hand_action": "chopsticks picking up food",
                            "expression": "satisfied, cheeks puffed",
                            "mood": "happy",
                            "caption_theme": "好饿啊终于开饭了",
                        },
                        {
                            "variation_id": "v2",
                            "description": "吃饱了",
                            "pose": "leaning back, patting stomach",
                            "body_action": "finished eating",
                            "hand_action": "hand on stomach",
                            "expression": "satisfied, full",
                            "mood": "content",
                            "caption_theme": "吃撑了...",
                        },
                    ],
                },
                # ========== 8. 20:00 - 晚间娱乐 ==========
                {
                    "activity_type": ActivityType.RELAXING,
                    "activity_description": "晚间休闲",
                    "activity_detail": "放松看看视频或玩游戏",
                    "location": "客厅",
                    "location_prompt": "living room, evening, cozy lighting",
                    "pose": "lounging on sofa",
                    "body_action": "relaxing at night",
                    "hand_action": "hands resting",
                    "expression": "relaxed, content",
                    "mood": "peaceful",
                    "outfit": "pajamas",
                    "accessories": "",
                    "environment": "cozy living room, dim lights",
                    "lighting": "warm evening light",
                    "weather_context": "night",
                    "caption_type": "monologue",
                    "suggested_caption_theme": "晚上的放松时光",
                    "scene_variations": [
                        {
                            "variation_id": "v1",
                            "description": "看视频",
                            "pose": "lying on sofa sideways",
                            "body_action": "watching entertainment",
                            "hand_action": "head propped on hand",
                            "expression": "amused, smiling",
                            "mood": "entertained",
                            "caption_theme": "哈哈哈笑死我了",
                        },
                        {
                            "variation_id": "v2",
                            "description": "发呆放空",
                            "pose": "lying flat on sofa",
                            "body_action": "staring at ceiling",
                            "hand_action": "hands on stomach",
                            "expression": "blank, peaceful",
                            "mood": "lazy",
                            "caption_theme": "什么都不想做",
                        },
                    ],
                },
                # ========== 9. 22:00 - 睡前护肤 ==========
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
                            "description": "准备睡觉",
                            "pose": "standing, yawning",
                            "body_action": "finished skincare",
                            "hand_action": "stretching arms",
                            "expression": "sleepy, eyes drooping",
                            "mood": "sleepy",
                            "caption_theme": "困了，晚安",
                        },
                    ],
                },
            ]
        else:
            # 工作日场景 - 共9个，对应9个时间点
            base = [
                # ========== 1. 07:30 - 起床 ==========
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
                            "pose": "reaching toward nightstand, turning off alarm",
                            "body_action": "still lying in bed",
                            "hand_action": "tapping alarm button",
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
                # ========== 2. 09:00 - 通勤 ==========
                {
                    "activity_type": ActivityType.COMMUTING,
                    "activity_description": "上班通勤",
                    "activity_detail": "出门上班啦",
                    "location": "地铁/公交",
                    "location_prompt": "subway, commuting, morning rush",
                    "pose": "standing, holding handrail",
                    "body_action": "commuting to work",
                    "hand_action": "holding handrail",
                    "expression": "sleepy but awake",
                    "mood": "neutral",
                    "outfit": "work clothes",
                    "accessories": "bag",
                    "environment": "crowded subway",
                    "lighting": "indoor lighting",
                    "weather_context": "morning",
                    "caption_type": "narrative",
                    "suggested_caption_theme": "通勤日常",
                    "scene_variations": [
                        {
                            "variation_id": "v1",
                            "description": "挤地铁",
                            "pose": "squeezed in crowd",
                            "body_action": "trying to keep balance",
                            "hand_action": "holding onto handrail tightly",
                            "expression": "slightly annoyed",
                            "mood": "tired",
                            "caption_theme": "早高峰真挤...",
                        },
                        {
                            "variation_id": "v2",
                            "description": "看窗外",
                            "pose": "leaning against door",
                            "body_action": "staring out window",
                            "hand_action": "hand on bag strap",
                            "expression": "blank, daydreaming",
                            "mood": "contemplative",
                            "caption_theme": "又是上班的一天",
                        },
                    ],
                },
                # ========== 3. 10:30 - 上午工作 ==========
                {
                    "activity_type": ActivityType.WORKING,
                    "activity_description": "上午工作",
                    "activity_detail": "到公司开始干活",
                    "location": "办公室",
                    "location_prompt": "office, desk, morning work",
                    "pose": "sitting at desk",
                    "body_action": "starting work",
                    "hand_action": "typing on keyboard",
                    "expression": "focused, awake",
                    "mood": "focused",
                    "outfit": "work clothes",
                    "accessories": "",
                    "environment": "modern office, morning",
                    "lighting": "office lighting",
                    "weather_context": "morning",
                    "caption_type": "narrative",
                    "suggested_caption_theme": "开始工作",
                    "scene_variations": [
                        {
                            "variation_id": "v1",
                            "description": "看邮件",
                            "pose": "leaning forward, reading screen",
                            "body_action": "checking emails",
                            "hand_action": "hand on mouse",
                            "expression": "concentrated, slight frown",
                            "mood": "focused",
                            "caption_theme": "邮件好多...",
                        },
                        {
                            "variation_id": "v2",
                            "description": "喝咖啡",
                            "pose": "sitting back, relaxed",
                            "body_action": "taking a coffee break",
                            "hand_action": "holding coffee cup",
                            "expression": "refreshed, slight smile",
                            "mood": "alert",
                            "caption_theme": "咖啡续命",
                        },
                    ],
                },
                # ========== 4. 12:00 - 午餐 ==========
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
                    ],
                },
                # ========== 5. 14:00 - 下午工作 ==========
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
                            "description": "开会",
                            "pose": "sitting in meeting room",
                            "body_action": "listening to presentation",
                            "hand_action": "taking notes",
                            "expression": "attentive, nodding",
                            "mood": "focused",
                            "caption_theme": "开会中...",
                        },
                    ],
                },
                # ========== 6. 16:00 - 下午休息 ==========
                {
                    "activity_type": ActivityType.RELAXING,
                    "activity_description": "下午茶休息",
                    "activity_detail": "工作累了休息一下",
                    "location": "办公室休息区",
                    "location_prompt": "office break room, afternoon",
                    "pose": "sitting in lounge chair",
                    "body_action": "taking a break",
                    "hand_action": "holding drink",
                    "expression": "tired but relaxed",
                    "mood": "tired",
                    "outfit": "work clothes",
                    "accessories": "",
                    "environment": "office break area",
                    "lighting": "natural afternoon light",
                    "weather_context": "afternoon",
                    "caption_type": "monologue",
                    "suggested_caption_theme": "摸鱼时间",
                    "scene_variations": [
                        {
                            "variation_id": "v1",
                            "description": "伸懒腰",
                            "pose": "stretching in chair",
                            "body_action": "taking a break, stretching",
                            "hand_action": "arms stretched overhead",
                            "expression": "tired, eyes closed",
                            "mood": "tired",
                            "caption_theme": "工作累了休息一下",
                        },
                        {
                            "variation_id": "v2",
                            "description": "喝咖啡",
                            "pose": "sitting back, relaxed",
                            "body_action": "enjoying coffee break",
                            "hand_action": "holding coffee cup",
                            "expression": "contemplative, relaxed",
                            "mood": "thoughtful",
                            "caption_theme": "下午茶时间",
                        },
                    ],
                },
                # ========== 7. 18:00 - 下班 ==========
                {
                    "activity_type": ActivityType.COMMUTING,
                    "activity_description": "下班啦",
                    "activity_detail": "终于下班可以回家了",
                    "location": "公司门口/地铁",
                    "location_prompt": "office entrance, evening, leaving work",
                    "pose": "walking out, relieved",
                    "body_action": "leaving work",
                    "hand_action": "carrying bag",
                    "expression": "relieved, happy",
                    "mood": "happy",
                    "outfit": "work clothes",
                    "accessories": "bag",
                    "environment": "evening city, sunset",
                    "lighting": "evening sunset light",
                    "weather_context": "evening",
                    "caption_type": "narrative",
                    "suggested_caption_theme": "下班快乐",
                    "scene_variations": [
                        {
                            "variation_id": "v1",
                            "description": "走出公司",
                            "pose": "walking, slight smile",
                            "body_action": "leaving office building",
                            "hand_action": "bag on shoulder",
                            "expression": "relieved, content",
                            "mood": "relieved",
                            "caption_theme": "终于下班了！",
                        },
                        {
                            "variation_id": "v2",
                            "description": "等地铁回家",
                            "pose": "standing on platform",
                            "body_action": "waiting for train",
                            "hand_action": "looking at time",
                            "expression": "tired but anticipating",
                            "mood": "anticipating",
                            "caption_theme": "回家回家",
                        },
                    ],
                },
                # ========== 8. 20:00 - 晚间休闲 ==========
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
                            "description": "看视频放松",
                            "pose": "lying on sofa",
                            "body_action": "watching entertainment",
                            "hand_action": "head propped on hand",
                            "expression": "amused, smiling",
                            "mood": "entertained",
                            "caption_theme": "追剧时间",
                        },
                    ],
                },
                # ========== 9. 22:00 - 睡前准备 ==========
                {
                    "activity_type": ActivityType.SELF_CARE,
                    "activity_description": "睡前准备",
                    "activity_detail": "洗漱准备睡觉",
                    "location": "浴室",
                    "location_prompt": "bathroom, night, getting ready for bed",
                    "pose": "standing at mirror",
                    "body_action": "doing night routine",
                    "hand_action": "brushing teeth",
                    "expression": "sleepy, tired",
                    "mood": "sleepy",
                    "outfit": "pajamas",
                    "accessories": "hair band",
                    "environment": "clean bathroom",
                    "lighting": "warm bathroom light",
                    "weather_context": "night",
                    "caption_type": "narrative",
                    "suggested_caption_theme": "睡前日常",
                    "scene_variations": [
                        {
                            "variation_id": "v1",
                            "description": "刷牙",
                            "pose": "standing at sink",
                            "body_action": "brushing teeth",
                            "hand_action": "holding toothbrush",
                            "expression": "sleepy, eyes half closed",
                            "mood": "tired",
                            "caption_theme": "困了...",
                        },
                        {
                            "variation_id": "v2",
                            "description": "护肤",
                            "pose": "leaning close to mirror",
                            "body_action": "applying skincare",
                            "hand_action": "patting face",
                            "expression": "focused, peaceful",
                            "mood": "peaceful",
                            "caption_theme": "睡前护肤很重要",
                        },
                    ],
                },
            ]

        # Phase 2：fallback 模板多套 - 生成变体模板集
        variant = copy.deepcopy(base)

        if is_holiday:
            # 假期变体：更偏户外/出门的周末感（避免全宅家重复）
            try:
                # 14:00 - 下午出门散步
                if len(variant) > 4:
                    variant[4].update(
                        {
                            "activity_type": ActivityType.EXERCISING,
                            "activity_description": "下午去公园散步",
                            "activity_detail": "天气不错，去附近公园走走放松一下",
                            "location": "公园",
                            "location_prompt": "city park, trees, afternoon, casual walk",
                            "pose": "walking on path, relaxed",
                            "body_action": "taking a walk, enjoying fresh air",
                            "hand_action": "hands behind back",
                            "expression": "relaxed smile",
                            "mood": "relaxed",
                            "outfit": "comfortable casual outfit",
                            "environment": "park path, greenery",
                            "lighting": "soft afternoon sunlight",
                            "caption_type": "share",
                            "suggested_caption_theme": "周末去公园透气",
                            "scene_variations": [
                                {
                                    "variation_id": "v1",
                                    "description": "闻花停一下",
                                    "pose": "standing near flowers, slight lean",
                                    "body_action": "smelling flowers",
                                    "hand_action": "hand near flowers",
                                    "expression": "gentle smile",
                                    "mood": "peaceful",
                                    "caption_theme": "花香好舒服",
                                },
                                {
                                    "variation_id": "v2",
                                    "description": "坐长椅休息",
                                    "pose": "sitting on bench, relaxed",
                                    "body_action": "resting, gazing into distance",
                                    "hand_action": "hands resting on lap",
                                    "expression": "dreamy",
                                    "mood": "relaxed",
                                    "caption_theme": "发呆也很治愈",
                                },
                            ],
                        }
                    )

                # 16:00 - 咖啡馆下午茶
                if len(variant) > 5:
                    variant[5].update(
                        {
                            "activity_type": ActivityType.HOBBY,
                            "activity_description": "咖啡馆下午茶",
                            "activity_detail": "找了家安静的咖啡馆坐坐",
                            "location": "咖啡馆",
                            "location_prompt": "cozy cafe, afternoon tea, warm atmosphere",
                            "pose": "sitting at table, relaxed",
                            "body_action": "enjoying coffee time",
                            "hand_action": "holding coffee cup",
                            "expression": "content, soft smile",
                            "mood": "peaceful",
                            "outfit": "casual dress",
                            "environment": "cafe interior, soft background",
                            "lighting": "warm indoor light",
                            "caption_type": "share",
                            "suggested_caption_theme": "周末咖啡时间",
                            "scene_variations": [
                                {
                                    "variation_id": "v1",
                                    "description": "搅拌咖啡",
                                    "pose": "leaning forward slightly",
                                    "body_action": "stirring coffee",
                                    "hand_action": "stirring with spoon",
                                    "expression": "focused, small smile",
                                    "mood": "relaxed",
                                    "caption_theme": "这杯香香的",
                                },
                                {
                                    "variation_id": "v2",
                                    "description": "看窗外发呆",
                                    "pose": "sitting by window",
                                    "body_action": "gazing outside",
                                    "hand_action": "hand supporting chin",
                                    "expression": "thoughtful",
                                    "mood": "contemplative",
                                    "caption_theme": "周末的时间过得好快",
                                },
                            ],
                        }
                    )

            except Exception:
                # 变体生成失败不影响 base 回退
                pass

        else:
            # 工作日变体：根据人设选择“学生日常”或“上班族另一套”
            try:
                if persona_mode == "student":
                    # 09:00 - 去学校
                    if len(variant) > 1:
                        variant[1].update(
                            {
                                "activity_type": ActivityType.COMMUTING,
                                "activity_description": "去学校上课",
                                "activity_detail": "背着包赶去学校，怕迟到",
                                "location": "地铁/公交",
                                "location_prompt": "subway, morning commute, backpack, campus vibe",
                                "pose": "standing, holding handrail",
                                "body_action": "commuting to campus",
                                "hand_action": "holding handrail",
                                "expression": "sleepy but determined",
                                "mood": "neutral",
                                "outfit": "casual student outfit, hoodie",
                                "accessories": "backpack",
                                "caption_type": "narrative",
                                "suggested_caption_theme": "早八赶路",
                            }
                        )

                    # 10:30 - 上课
                    if len(variant) > 2:
                        variant[2].update(
                            {
                                "activity_type": ActivityType.STUDYING,
                                "activity_description": "上午上课",
                                "activity_detail": "坐在教室里认真听课记笔记",
                                "location": "教室",
                                "location_prompt": "classroom, lecture, daylight",
                                "pose": "sitting at desk",
                                "body_action": "listening to lecture",
                                "hand_action": "taking notes",
                                "expression": "attentive",
                                "mood": "focused",
                                "outfit": "casual student outfit",
                                "environment": "classroom, desks",
                                "lighting": "natural daylight",
                                "caption_type": "narrative",
                                "suggested_caption_theme": "课堂日常",
                                "scene_variations": [
                                    {
                                        "variation_id": "v1",
                                        "description": "认真记笔记",
                                        "pose": "leaning forward",
                                        "body_action": "writing notes",
                                        "hand_action": "holding pen",
                                        "expression": "concentrated",
                                        "mood": "focused",
                                        "caption_theme": "记不完的笔记",
                                    },
                                    {
                                        "variation_id": "v2",
                                        "description": "听课发呆一下",
                                        "pose": "sitting upright",
                                        "body_action": "zoning out briefly",
                                        "hand_action": "hand supporting cheek",
                                        "expression": "blank stare",
                                        "mood": "tired",
                                        "caption_theme": "需要补眠",
                                    },
                                ],
                            }
                        )

                    # 12:00 - 食堂午餐
                    if len(variant) > 3:
                        variant[3].update(
                            {
                                "activity_type": ActivityType.EATING,
                                "activity_description": "食堂午餐",
                                "activity_detail": "下课去食堂吃饭",
                                "location": "学校食堂",
                                "location_prompt": "school cafeteria, lunch time",
                                "pose": "sitting at table",
                                "body_action": "having lunch",
                                "hand_action": "holding chopsticks",
                                "expression": "happy",
                                "mood": "happy",
                                "outfit": "casual student outfit",
                                "caption_type": "share",
                                "suggested_caption_theme": "今天食堂吃啥",
                            }
                        )

                    # 14:00 - 图书馆自习
                    if len(variant) > 4:
                        variant[4].update(
                            {
                                "activity_type": ActivityType.STUDYING,
                                "activity_description": "图书馆自习",
                                "activity_detail": "下午去图书馆坐一会儿，写作业/复习",
                                "location": "图书馆",
                                "location_prompt": "library, study desk, quiet atmosphere",
                                "pose": "sitting at desk",
                                "body_action": "studying, reading",
                                "hand_action": "holding pen",
                                "expression": "focused",
                                "mood": "focused",
                                "outfit": "casual student outfit",
                                "caption_type": "monologue",
                                "suggested_caption_theme": "自习加油",
                            }
                        )

                    # 16:00 - 小憩/咖啡补能
                    if len(variant) > 5:
                        variant[5].update(
                            {
                                "activity_type": ActivityType.RELAXING,
                                "activity_description": "课间休息",
                                "activity_detail": "学习久了去喝点东西歇一下",
                                "location": "校园咖啡角",
                                "location_prompt": "campus cafe corner, afternoon, cozy",
                                "pose": "sitting, relaxed",
                                "body_action": "taking a short break",
                                "hand_action": "holding drink",
                                "expression": "relieved",
                                "mood": "relaxed",
                                "outfit": "casual student outfit",
                                "caption_type": "share",
                                "suggested_caption_theme": "补充能量",
                            }
                        )

                    # 18:00 - 回宿舍
                    if len(variant) > 6:
                        variant[6].update(
                            {
                                "activity_type": ActivityType.COMMUTING,
                                "activity_description": "回宿舍",
                                "activity_detail": "一天的课/自习结束，慢慢走回宿舍",
                                "location": "校园路上",
                                "location_prompt": "campus, sunset, walking",
                                "pose": "walking, relaxed",
                                "body_action": "walking back",
                                "hand_action": "carrying bag",
                                "expression": "tired but content",
                                "mood": "relaxed",
                                "outfit": "casual student outfit",
                                "caption_type": "narrative",
                                "suggested_caption_theme": "回去休息",
                            }
                        )

                    # 20:00 - 宿舍放松
                    if len(variant) > 7:
                        variant[7].update(
                            {
                                "activity_type": ActivityType.RELAXING,
                                "activity_description": "宿舍放松",
                                "activity_detail": "回到宿舍，躺一会儿放松一下",
                                "location": "宿舍",
                                "location_prompt": "dorm room, cozy, evening",
                                "pose": "lounging on bed",
                                "body_action": "relaxing",
                                "hand_action": "head propped on hand",
                                "expression": "relieved",
                                "mood": "relaxed",
                                "outfit": "comfortable home clothes",
                                "caption_type": "monologue",
                                "suggested_caption_theme": "终于可以休息了",
                            }
                        )

                else:
                    # 上班族另一套：把部分节点换成更常见的办公日常，降低跨天重复感
                    if len(variant) > 3:
                        variant[3]["activity_detail"] = "中午出来吃点清淡的，顺便透透气"
                        variant[3]["suggested_caption_theme"] = "午餐小确幸"
                    if len(variant) > 5:
                        variant[5]["activity_description"] = "下午茶摸鱼"
                        variant[5]["activity_detail"] = "工作到下午有点累，喝点东西缓缓"
                        variant[5]["caption_type"] = "share"
                        variant[5]["suggested_caption_theme"] = "下午茶回血"
                    if len(variant) > 7:
                        variant[7]["activity_description"] = "回家做饭"
                        variant[7]["activity_detail"] = "下班回家简单做点吃的，放松一下"
                        variant[7]["location"] = "家里厨房"
                        variant[7]["location_prompt"] = "home kitchen, evening, warm light"
                        variant[7]["hand_action"] = "holding cooking utensil"
                        variant[7]["caption_type"] = "share"
                        variant[7]["suggested_caption_theme"] = "今晚吃点什么"

            except Exception:
                pass

        candidate_sets: List[List[Dict[str, Any]]] = [base, variant]
        select_key = f"{date}|{persona_mode}|{'holiday' if is_holiday else 'workday'}"
        selected_index = int(hashlib.md5(select_key.encode("utf-8")).hexdigest(), 16) % len(candidate_sets)
        logger.info(
            "选择回退模板集: "
            f"mode={'holiday' if is_holiday else 'workday'}, persona={persona_mode}, "
            f"set={selected_index + 1}/{len(candidate_sets)}"
        )
        return candidate_sets[selected_index]

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
        """清理旧的日程文件。

        Phase 2：跨天去重需要保留最近 7 天日程用于回灌 prompt，因此不再删除所有非当天文件。
        默认保留窗口：current_date-7 ... current_date（含当天，共 8 天文件）。

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

            try:
                base_date = datetime.strptime(current_date, "%Y-%m-%d")
            except ValueError:
                logger.warning(f"current_date 格式不正确，跳过清理: {current_date}")
                return

            # N 表示“向前保留 N 天”（不含未来限制），为了支持回灌最近7天摘要，默认 N=7。
            retention_days_raw = self.plugin.get_config("auto_selfie.schedule_retention_days", 7)
            try:
                retention_days = int(retention_days_raw)
            except (TypeError, ValueError):
                retention_days = 7
            retention_days = max(0, retention_days)

            keep_from = base_date - timedelta(days=retention_days)

            deleted_count = 0
            for file_path in schedule_files:
                filename = os.path.basename(file_path)
                if not (filename.startswith("daily_schedule_") and filename.endswith(".json")):
                    continue

                file_date_str = filename[15:-5]  # daily_schedule_YYYY-MM-DD.json
                try:
                    file_date = datetime.strptime(file_date_str, "%Y-%m-%d")
                except ValueError:
                    # 文件名不符合日期格式，保守起见不删除
                    continue

                # 删除早于 keep_from 的文件；保留 keep_from 及之后的文件（包含当天与最近 N 天）
                if file_date < keep_from:
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                        logger.debug(f"已删除旧日程文件: {filename}")
                    except OSError as e:
                        logger.warning(f"删除旧日程文件失败 {filename}: {e}")

            if deleted_count > 0:
                logger.info(f"已清理 {deleted_count} 个旧日程文件（保留最近 {retention_days} 天窗口）")

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
                current_signature = self._get_schedule_persona_signature()
                if existing.character_persona == current_signature:
                    logger.info(f"从文件加载已有日程: {date}")
                    return existing

                logger.info(
                    "检测到日程人设配置变更，触发重新生成 "
                    f"(old_signature_len={len(existing.character_persona)}, new_signature_len={len(current_signature)})"
                )

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
