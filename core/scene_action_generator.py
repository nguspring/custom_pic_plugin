"""
场景动作生成器模块

根据日程条目(ScheduleEntry)生成符合情境的动作和 Stable Diffusion 提示词。
这是动态日程系统的核心组件之一。
"""

import random
import re
from typing import Any, Dict, List, Optional

from src.common.logger import get_logger

from .schedule_models import ActivityType, ScheduleEntry, SceneVariation

logger = get_logger("SceneActionGenerator")


class SceneActionGenerator:
    """
    场景动作生成器 - 根据场景生成符合情境的动作

    负责将 ScheduleEntry 转换为自拍所需的动作描述和 SD 提示词。
    支持多层次回退机制，确保始终能生成有效的提示词。
    """

    # 活动类型到动作的映射
    # 每个活动类型包含多个可选动作，系统会随机选择或使用 LLM 生成的动作
    ACTIVITY_ACTIONS: Dict[str, List[str]] = {
        "sleeping": [
            "lying down, relaxed",
            "hugging pillow, cozy",
            "stretching arms, sleepy",
            "curled up in bed",
            "peaceful sleeping pose",
        ],
        "waking_up": [
            "stretching, yawning",
            "rubbing eyes, sleepy",
            "sitting on bed edge",
            "messy hair, just woke up",
            "holding alarm clock",
        ],
        "eating": [
            "holding chopsticks, eating",
            "holding cup, drinking",
            "picking up food",
            "holding fork and knife",
            "holding spoon, tasting",
            "holding bowl, eating",
        ],
        "working": [
            "typing on laptop",
            "writing notes",
            "looking at screen, focused",
            "holding pen, thinking",
            "reading documents",
            "reviewing notes, professional",
        ],
        "studying": [
            "holding book, reading",
            "writing in notebook",
            "looking at textbook",
            "holding pen, studying",
            "taking notes",
            "highlighting text",
        ],
        "exercising": [
            "stretching, athletic",
            "holding water bottle",
            "wiping sweat, tired",
            "doing yoga pose",
            "running pose",
            "holding dumbbells",
        ],
        "relaxing": [
            "lying on couch, relaxed",
            "resting head on hand, zoning out",
            "playing with pet",
            "reading magazine",
            "listening to music, headphones",
            "watching TV, relaxed",
        ],
        "socializing": [
            "waving hand, greeting",
            "making peace sign, happy",
            "laughing, joyful",
            "talking with friends",
            "taking group photo",
            "cheering, celebration",
        ],
        "commuting": [
            "holding bag, walking",
            "standing, waiting",
            "looking out window, daydreaming",
            "wearing earbuds, commuting",
            "holding coffee, on the go",
            "checking watch, hurried",
        ],
        "hobby": [
            "holding camera, taking photos",
            "playing instrument",
            "crafting, creative",
            "painting, artistic",
            "playing video games",
            "gardening, nature",
        ],
        "self_care": [
            "applying makeup, mirror",
            "brushing hair",
            "holding mirror, checking",
            "skincare routine",
            "nail painting",
            "face mask, relaxing",
        ],
        "other": [
            "standing, casual pose",
            "sitting, relaxed",
            "casual pose, natural",
            "looking at camera",
            "peace sign, friendly",
        ],
    }

    # 通用的自拍手部动作库（作为回退）
    FALLBACK_HAND_ACTIONS: List[str] = [
        "peace sign, v sign",
        "thumbs up, positive gesture",
        "ok sign, hand gesture",
        "waving hand, greeting",
        "touching own cheek gently",
        "finger heart, cute pose",
        "hand on hip, confident",
        "playing with hair",
        "framing face with hand, selfie pose",
    ]

    def __init__(self, config_provider: Any):
        """
        初始化生成器

        Args:
            config_provider: 具有 get_config 方法的对象（插件实例或 Action 实例）
        """
        self._config_provider = config_provider
        logger.info("SceneActionGenerator 初始化完成")

    def get_config(self, key: str, default: Any = None) -> Any:
        """
        获取配置值

        Args:
            key: 配置键
            default: 默认值

        Returns:
            配置值
        """
        if hasattr(self._config_provider, "get_config"):
            return self._config_provider.get_config(key, default)
        return default

    def get_action_for_scene(self, schedule_entry: ScheduleEntry) -> Dict[str, str]:
        """
        根据场景获取适合的动作

        优先级：
        1. 使用 ScheduleEntry 中 LLM 生成的动作（如果存在）
        2. 根据 activity_type 从映射中选择合适的动作
        3. 使用通用手部动作作为回退

        Args:
            schedule_entry: 日程条目

        Returns:
            包含动作信息的字典:
            {
                "hand_action": "手部动作描述",
                "body_action": "身体动作描述",
                "pose": "姿势描述"
            }
        """
        result = {
            "hand_action": "",
            "body_action": "",
            "pose": "",
        }

        # 1. 优先使用 ScheduleEntry 中的 LLM 生成动作
        if schedule_entry.hand_action and schedule_entry.hand_action.strip():
            result["hand_action"] = schedule_entry.hand_action.strip()
            logger.debug(f"使用 LLM 生成的手部动作: {result['hand_action']}")

        if schedule_entry.body_action and schedule_entry.body_action.strip():
            result["body_action"] = schedule_entry.body_action.strip()
            logger.debug(f"使用 LLM 生成的身体动作: {result['body_action']}")

        if schedule_entry.pose and schedule_entry.pose.strip():
            result["pose"] = schedule_entry.pose.strip()
            logger.debug(f"使用 LLM 生成的姿势: {result['pose']}")

        # 2. 如果手部动作为空，根据 activity_type 选择
        if not result["hand_action"]:
            activity_type = schedule_entry.activity_type
            if isinstance(activity_type, ActivityType):
                activity_key = activity_type.value
            else:
                activity_key = str(activity_type)

            # 获取该活动类型的动作列表
            actions = self.ACTIVITY_ACTIONS.get(activity_key, self.ACTIVITY_ACTIONS["other"])

            if actions:
                selected_action = random.choice(actions)
                result["hand_action"] = selected_action
                logger.debug(f"从活动类型 '{activity_key}' 选择动作: {selected_action}")

        # 3. 如果仍然为空，使用通用回退
        if not result["hand_action"]:
            result["hand_action"] = random.choice(self.FALLBACK_HAND_ACTIONS)
            logger.debug(f"使用回退手部动作: {result['hand_action']}")

        return result

    def convert_to_sd_prompt(
        self,
        schedule_entry: ScheduleEntry,
        selfie_style: str = "standard",
        scene_variation: Optional[SceneVariation] = None,
    ) -> str:
        """
        将场景转换为 Stable Diffusion 提示词

        生成完整的自拍提示词，包含：
        - 强制主体设置
        - 外观（从配置读取）
        - 服装和配饰
        - 姿势和动作
        - 表情
        - 地点和环境
        - 光线
        - 自拍视角

        Args:
            schedule_entry: 日程条目
            selfie_style: 自拍风格 ("standard" 或 "mirror")

        Returns:
            完整的 SD 提示词字符串
        """
        prompt_parts: List[str] = []

        # 1. 强制主体设置
        forced_subject = "(1girl:1.4), (solo:1.3)"
        prompt_parts.append(forced_subject)

        # 2. 获取角色外观（从配置）
        bot_appearance = str(self.get_config("selfie.prompt_prefix", "")).strip()
        if bot_appearance:
            prompt_parts.append(bot_appearance)

        # 3. 表情（变体优先）
        expression_src = ""
        if scene_variation and scene_variation.expression and scene_variation.expression.strip():
            expression_src = scene_variation.expression.strip()
        elif schedule_entry.expression and schedule_entry.expression.strip():
            expression_src = schedule_entry.expression.strip()

        if expression_src:
            prompt_parts.append(f"({expression_src}:1.2)")

        # 4. 获取动作
        actions = self.get_action_for_scene(schedule_entry)

        # 姿势（变体优先）
        if scene_variation and scene_variation.pose and scene_variation.pose.strip():
            prompt_parts.append(scene_variation.pose.strip())
        elif actions["pose"]:
            prompt_parts.append(actions["pose"])
        elif schedule_entry.pose:
            prompt_parts.append(schedule_entry.pose)

        # 身体动作（变体优先）
        body_action_src = ""
        if scene_variation and scene_variation.body_action and scene_variation.body_action.strip():
            body_action_src = scene_variation.body_action.strip()
        elif actions["body_action"]:
            body_action_src = actions["body_action"]
        elif schedule_entry.body_action and schedule_entry.body_action.strip():
            body_action_src = schedule_entry.body_action.strip()

        if body_action_src:
            prompt_parts.append(body_action_src)

        # 手部动作（变体优先）
        hand_action_src = ""
        if scene_variation and scene_variation.hand_action and scene_variation.hand_action.strip():
            hand_action_src = scene_variation.hand_action.strip()
        elif schedule_entry.hand_action and schedule_entry.hand_action.strip():
            hand_action_src = schedule_entry.hand_action.strip()
        elif actions["hand_action"]:
            hand_action_src = actions["hand_action"]

        # standard 自拍严格禁止手机类词汇（mirror 模式允许）
        if selfie_style == "standard" and hand_action_src:
            if re.search(r"\b(phone|smartphone|mobile|device)\b", hand_action_src, flags=re.IGNORECASE):
                hand_action_src = "resting head on hand"

        if hand_action_src:
            # 根据自拍风格调整手部动作描述
            if selfie_style == "standard":
                # 前置自拍：强调单手动作，避免双手
                hand_prompt = (
                    f"(visible free hand {hand_action_src}:1.4), "
                    "(only one hand visible in frame:1.5), "
                    "(single hand gesture:1.3)"
                )
            else:
                # 对镜自拍：可以看到两只手
                hand_prompt = f"({hand_action_src}:1.3)"
            prompt_parts.append(hand_prompt)

        # 5. 服装
        if schedule_entry.outfit and schedule_entry.outfit.strip():
            prompt_parts.append(schedule_entry.outfit.strip())

        # 6. 配饰
        if schedule_entry.accessories and schedule_entry.accessories.strip():
            prompt_parts.append(schedule_entry.accessories.strip())

        # 7. 地点
        if schedule_entry.location_prompt and schedule_entry.location_prompt.strip():
            prompt_parts.append(schedule_entry.location_prompt.strip())

        # 8. 环境
        if schedule_entry.environment and schedule_entry.environment.strip():
            prompt_parts.append(schedule_entry.environment.strip())

        # 9. 光线
        if schedule_entry.lighting and schedule_entry.lighting.strip():
            prompt_parts.append(schedule_entry.lighting.strip())

        # 10. 天气上下文（如果有）
        if schedule_entry.weather_context and schedule_entry.weather_context.strip():
            prompt_parts.append(schedule_entry.weather_context.strip())

        # 11. 自拍风格特定设置
        if selfie_style == "mirror":
            selfie_scene = (
                "mirror selfie, reflection in mirror, "
                "holding phone in hand, phone visible, "
                "looking at mirror, indoor scene"
            )
        else:
            selfie_scene = (
                "selfie, front camera view, POV selfie, "
                "(front facing selfie camera angle:1.3), "
                "looking at camera, slight high angle selfie, "
                "upper body shot, cowboy shot, "
                "(centered composition:1.2)"
            )
        prompt_parts.append(selfie_scene)

        # 12. 过滤空值并拼接
        prompt_parts = [p for p in prompt_parts if p and p.strip()]
        final_prompt = ", ".join(prompt_parts)

        # 13. 去重
        keywords = [kw.strip() for kw in final_prompt.split(",")]
        seen = set()
        unique_keywords = []
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower not in seen and kw:
                seen.add(kw_lower)
                unique_keywords.append(kw)

        final_prompt = ", ".join(unique_keywords)

        logger.info(f"场景动作生成器生成的提示词: {final_prompt[:200]}...")
        return final_prompt

    def get_negative_prompt_for_style(self, selfie_style: str) -> str:
        """
        获取指定自拍风格的负面提示词

        Args:
            selfie_style: 自拍风格

        Returns:
            负面提示词字符串
        """
        # 基础负面提示词
        base_negative = str(self.get_config("selfie.negative_prompt", "")).strip()

        if selfie_style == "standard":
            # 前置自拍需要额外的防双手负面词
            anti_dual_hands = (
                "two phones, camera in both hands, "
                "holding phone with both hands, "
                "extra hands, extra arms, 3 hands, 4 hands, "
                "multiple hands, both hands holding phone, "
                "phone in frame, visible phone in hand, "
                "phone screen visible, floating phone, "
                "both hands visible, two hands making gesture, "
                "symmetrical hands, mirrored hands, "
                "hand at edge of frame, partial hand visible at edge"
            )
            if base_negative:
                return f"{base_negative}, {anti_dual_hands}"
            return anti_dual_hands

        return base_negative

    def create_caption_context(self, schedule_entry: ScheduleEntry) -> Dict[str, str]:
        """
        创建配文生成的上下文信息

        Args:
            schedule_entry: 日程条目

        Returns:
            包含配文上下文的字典
        """
        return {
            "activity_description": schedule_entry.activity_description,
            "activity_detail": schedule_entry.activity_detail,
            "location": schedule_entry.location,
            "mood": schedule_entry.mood,
            "time_point": schedule_entry.time_point,
            "caption_type": schedule_entry.caption_type,
            "suggested_theme": schedule_entry.suggested_caption_theme,
        }
