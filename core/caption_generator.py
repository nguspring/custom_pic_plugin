"""
配文生成器模块

负责根据配文类型、场景上下文使用 LLM 生成自然的配文。
支持 5 种配文类型：叙事式、询问式、分享式、独白式、无配文。

v3.6.0: 移除对 NarrativeScene 的依赖，改用 Any 类型注解
"""

import random
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.common.logger import get_logger

from .selfie_models import (
    CaptionType,
    CaptionWeightConfig,
)

logger = get_logger("CaptionGenerator")


class CaptionGenerator:
    """配文生成器

    负责根据配文类型、场景上下文使用 LLM 生成自然的配文。
    支持 5 种配文类型：叙事式、询问式、分享式、独白式、无配文。
    """

    # Prompt 模板定义
    PROMPT_TEMPLATES: Dict[CaptionType, str] = {
        CaptionType.NARRATIVE: """你正在模拟一个真人发自拍的场景。
当前场景：{scene_description}
之前的动态：
{narrative_context}

请生成一句延续日常故事线的配文，要有承上启下的感觉。
要求：
1. 语气自然，像真人朋友圈/QQ空间的说说
2. 可以提及之前的场景，形成连贯感
3. 长度15-30字
4. 当前情绪：{mood}
5. 直接输出配文，不要任何解释

配文：""",
        CaptionType.ASK: """你正在模拟一个真人发自拍征求意见。
当前场景：{scene_description}

请生成一句询问式配文，期待朋友们的互动。
要求：
1. 语气俏皮可爱
2. 询问对方对照片/穿搭/状态的看法
3. 长度10-20字
4. 当前情绪：{mood}
5. 直接输出配文，不要任何解释

配文：""",
        CaptionType.SHARE: """你正在模拟一个真人分享日常状态。
当前场景：{scene_description}

请生成一句分享式配文，单纯分享心情。
要求：
1. 不期待回应，纯粹分享
2. 可以感叹天气、心情、环境
3. 长度10-25字
4. 当前情绪：{mood}
5. 直接输出配文，不要任何解释

配文：""",
        CaptionType.MONOLOGUE: """你正在模拟一个真人的自言自语。
当前情绪：{mood}

请生成一句独白式配文，像是自言自语、碎碎念。
要求：
1. 简短，像是随口说的
2. 可以是"好困""好无聊""饿了"这种
3. 长度5-15字
4. 直接输出配文，不要任何解释

配文：""",
    }

    # 各类型的备用配文列表
    FALLBACK_CAPTIONS: Dict[CaptionType, List[str]] = {
        CaptionType.NARRATIVE: [
            "新的一天开始啦~",
            "继续今天的日常",
            "时间过得真快呀",
            "又是充实的一天",
            "记录一下此刻",
        ],
        CaptionType.ASK: [
            "今天的状态怎么样？",
            "这样穿搭可以吗？",
            "猜猜我在干嘛~",
            "给点意见呗？",
            "你们觉得呢？",
        ],
        CaptionType.SHARE: [
            "今天心情不错呢",
            "天气真好~",
            "享受这一刻",
            "平平淡淡的日常",
            "简简单单的幸福",
        ],
        CaptionType.MONOLOGUE: [
            "好困...",
            "饿了",
            "好无聊啊",
            "嘿嘿",
            "呜呜",
            "哼",
            "嗯...",
        ],
        CaptionType.NONE: [],
    }

    def __init__(self, plugin_instance: Any):
        """初始化生成器

        Args:
            plugin_instance: 插件实例，用于读取配置和调用 LLM API
        """
        self.plugin = plugin_instance
        logger.info("CaptionGenerator 初始化完成")

    # ==================== 人设注入 ====================

    def _get_persona_block(self) -> str:
        """获取人设配置并构建人设提示块

        根据用户配置的人设描述和表达风格，构建注入到 prompt 前的人设块。

        Returns:
            构建好的人设提示块字符串，如果未启用则返回空字符串
        """
        # 检查是否启用人设注入
        persona_enabled = self.plugin.get_config("auto_selfie.caption_persona_enabled", True)

        if not persona_enabled:
            logger.debug("人设注入未启用")
            return ""

        # 获取人设配置
        persona_text = self.plugin.get_config("auto_selfie.caption_persona_text", "是一个喜欢分享日常的女生")
        reply_style = self.plugin.get_config("auto_selfie.caption_reply_style", "语气自然，符合年轻人社交风格")

        # 如果两个配置都为空，返回空字符串
        if not persona_text and not reply_style:
            logger.debug("人设和风格配置均为空，跳过注入")
            return ""

        # 构建人设块
        persona_block_parts = []

        if persona_text:
            persona_block_parts.append(f"【你的人设】\n你{persona_text}")

        if reply_style:
            persona_block_parts.append(f"【表达风格】\n{reply_style}")

        persona_block = "\n\n".join(persona_block_parts)

        logger.info(f"[DEBUG-叙事] 人设注入已启用，人设块长度: {len(persona_block)}")
        logger.debug(f"人设块内容: {persona_block[:100]}...")

        return persona_block + "\n\n"

    # ==================== 配文类型选择 ====================

    def select_caption_type(
        self,
        scene: Optional[Any] = None,
        narrative_context: str = "",
        current_hour: Optional[int] = None,
    ) -> CaptionType:
        """智能选择配文类型

        Args:
            scene: 当前场景（如果有，可以是 ScheduleEntry 或任何有 caption_type 属性的对象）
            narrative_context: 叙事上下文
            current_hour: 当前小时（0-23）

        Returns:
            选择的配文类型

        选择逻辑：
        1. 如果场景指定了配文类型，优先使用
        2. 否则根据时间段权重随机选择
        """
        logger.info("[DEBUG-叙事] select_caption_type() 被调用")
        scene_id = getattr(scene, "scene_id", None) or getattr(scene, "time_point", None)
        logger.info(
            f"[DEBUG-叙事] 参数 - scene: {scene_id if scene else 'None'}, context长度: {len(narrative_context)}, hour: {current_hour}"
        )

        # 如果场景指定了配文类型，优先使用
        if scene is not None and hasattr(scene, "caption_type") and scene.caption_type:
            caption_type_value = scene.caption_type if isinstance(scene.caption_type, str) else scene.caption_type.value
            logger.info(f"[DEBUG-叙事] 使用场景指定的配文类型: {caption_type_value}")
            logger.debug(f"使用场景指定的配文类型: {caption_type_value}")
            # 如果是字符串，转换为枚举
            if isinstance(scene.caption_type, str):
                try:
                    return CaptionType(scene.caption_type)
                except ValueError:
                    pass  # 无效的类型，使用默认逻辑
            else:
                return scene.caption_type

        # 获取当前小时
        if current_hour is None:
            current_hour = datetime.now().hour

        # 根据时间段获取权重配置
        weight_config = CaptionWeightConfig.for_time_period(current_hour)
        weights = weight_config.get_weights_list()
        logger.info(f"[DEBUG-叙事] 时间段权重配置 (hour={current_hour}): {weights}")

        # 获取所有配文类型（按枚举顺序）
        caption_types = list(CaptionType)
        logger.debug(f"[DEBUG-叙事] 可选配文类型: {[t.value for t in caption_types]}")

        # 随机选择
        selected_type = random.choices(caption_types, weights=weights, k=1)[0]
        logger.info(f"[DEBUG-叙事] 随机选择的配文类型: {selected_type.value}")
        logger.debug(f"根据时间段 {current_hour}:00 权重选择配文类型: {selected_type.value}")

        return selected_type

    # ==================== 配文生成主方法 ====================

    async def generate_caption(
        self,
        caption_type: CaptionType,
        scene_description: str = "",
        narrative_context: str = "",
        image_prompt: str = "",
        mood: str = "neutral",
        visual_summary: str = "",
    ) -> str:
        """生成配文

        Args:
            caption_type: 配文类型
            scene_description: 场景描述（中文）
            narrative_context: 叙事上下文
            image_prompt: 图片提示词（英文，用于参考）
            mood: 当前情绪
            visual_summary: 图片的视觉摘要（VLM 输出，中文）。用于保证“配文贴图”。

        Returns:
            生成的配文，如果类型是 NONE 则返回空字符串
        """
        logger.info("[DEBUG-叙事] generate_caption() 被调用")
        logger.info(f"[DEBUG-叙事] 参数 - type: {caption_type.value}, scene: {scene_description}, mood: {mood}")
        logger.info(
            f"[DEBUG-叙事] 参数 - context长度: {len(narrative_context)}, image_prompt: {image_prompt[:50] if image_prompt else 'None'}..."
        )

        # 如果是无配文类型，直接返回空字符串
        if caption_type == CaptionType.NONE:
            logger.info("[DEBUG-叙事] 配文类型为 NONE，返回空字符串")
            logger.debug("配文类型为 NONE，返回空字符串")
            return ""

        logger.info(f"开始生成配文，类型: {caption_type.value}, 场景: {scene_description}")

        try:
            # 根据类型调用对应的生成方法
            logger.info(f"[DEBUG-叙事] 调用 _{caption_type.value} 配文生成方法...")
            if caption_type == CaptionType.NARRATIVE:
                caption = await self._generate_narrative_caption(
                    scene_description, narrative_context, mood, visual_summary
                )
            elif caption_type == CaptionType.ASK:
                caption = await self._generate_ask_caption(scene_description, mood, visual_summary)
            elif caption_type == CaptionType.SHARE:
                caption = await self._generate_share_caption(scene_description, mood, visual_summary)
            elif caption_type == CaptionType.MONOLOGUE:
                caption = await self._generate_monologue_caption(mood, visual_summary)
            else:
                logger.warning(f"[DEBUG-叙事] 未知的配文类型: {caption_type}")
                logger.warning(f"未知的配文类型: {caption_type}")
                caption = ""

            # 如果生成失败，使用备用配文
            if not caption:
                logger.warning("[DEBUG-叙事] 配文生成返回空，使用备用配文")
                logger.warning("配文生成失败，使用备用配文")
                caption = self._get_fallback_caption(caption_type)
                logger.info(f"[DEBUG-叙事] 备用配文: {caption}")

            logger.info(f"[DEBUG-叙事] 配文生成完成: {caption}")
            logger.info(f"配文生成完成: {caption}")
            return caption

        except Exception as e:
            import traceback

            logger.error(f"[DEBUG-叙事] 配文生成异常: {e}")
            logger.error(f"[DEBUG-叙事] 异常堆栈: {traceback.format_exc()}")
            logger.error(f"配文生成异常: {e}")
            fallback = self._get_fallback_caption(caption_type)
            logger.info(f"[DEBUG-叙事] 异常回退配文: {fallback}")
            return fallback

    # ==================== 各类型专用生成方法 ====================

    def _get_visual_block(self, visual_summary: str) -> str:
        """构建视觉摘要提示块。

        Phase 4：用于“配文贴图”。视觉摘要来自 VLM，对应本次实际生成的图片。
        """
        if not visual_summary or not visual_summary.strip():
            return ""

        # 强约束：配文应以视觉摘要为准，避免编造画面外元素。
        return (
            "【图片视觉摘要（以此为准）】\n"
            f"{visual_summary.strip()}\n\n"
            "写配文时：\n"
            "- 不要提及摘要里没有的具体物品/动作\n"
            "- 不要出现 phone/smartphone/mobile/device 等手机相关词（standard 自拍手机在画面外）\n\n"
        )

    async def _generate_narrative_caption(
        self,
        scene_description: str,
        narrative_context: str,
        mood: str,
        visual_summary: str,
    ) -> str:
        """生成叙事式配文

        特点：延续日常故事线，有承上启下感

        Args:
            scene_description: 场景描述
            narrative_context: 叙事上下文
            mood: 当前情绪

        Returns:
            生成的叙事式配文
        """
        # 如果没有叙事上下文，提供默认值
        if not narrative_context:
            narrative_context = "今天还没有发过自拍。"

        # 获取人设块并注入到 prompt 前面
        persona_block = self._get_persona_block()
        visual_block = self._get_visual_block(visual_summary)

        prompt = (
            persona_block
            + visual_block
            + self.PROMPT_TEMPLATES[CaptionType.NARRATIVE].format(
                scene_description=scene_description or "日常",
                narrative_context=narrative_context,
                mood=mood,
            )
        )

        return await self._call_llm(prompt)

    async def _generate_ask_caption(
        self,
        scene_description: str,
        mood: str,
        visual_summary: str,
    ) -> str:
        """生成询问式配文

        特点：征求意见，期待互动

        Args:
            scene_description: 场景描述
            mood: 当前情绪

        Returns:
            生成的询问式配文
        """
        # 获取人设块并注入到 prompt 前面
        persona_block = self._get_persona_block()
        visual_block = self._get_visual_block(visual_summary)

        prompt = (
            persona_block
            + visual_block
            + self.PROMPT_TEMPLATES[CaptionType.ASK].format(
                scene_description=scene_description or "自拍",
                mood=mood,
            )
        )

        return await self._call_llm(prompt)

    async def _generate_share_caption(
        self,
        scene_description: str,
        mood: str,
        visual_summary: str,
    ) -> str:
        """生成分享式配文

        特点：分享心情/状态，不期待回应

        Args:
            scene_description: 场景描述
            mood: 当前情绪

        Returns:
            生成的分享式配文
        """
        # 获取人设块并注入到 prompt 前面
        persona_block = self._get_persona_block()
        visual_block = self._get_visual_block(visual_summary)

        prompt = (
            persona_block
            + visual_block
            + self.PROMPT_TEMPLATES[CaptionType.SHARE].format(
                scene_description=scene_description or "日常",
                mood=mood,
            )
        )

        return await self._call_llm(prompt)

    async def _generate_monologue_caption(
        self,
        mood: str,
        visual_summary: str,
    ) -> str:
        """生成独白式配文

        特点：自言自语，碎碎念

        Args:
            mood: 当前情绪

        Returns:
            生成的独白式配文
        """
        # 获取人设块并注入到 prompt 前面
        persona_block = self._get_persona_block()
        visual_block = self._get_visual_block(visual_summary)

        prompt = (
            persona_block
            + visual_block
            + self.PROMPT_TEMPLATES[CaptionType.MONOLOGUE].format(
                mood=mood,
            )
        )

        return await self._call_llm(prompt)

    # ==================== LLM 调用封装 ====================

    async def _call_llm(self, prompt: str) -> str:
        """调用 LLM 生成内容

        使用 MaiBot 的 llm_api 进行调用。
        默认使用 MaiBot 主配置中的 replyer（回复模型），这样配文风格与麦麦的回复一致。
        如果用户配置了自定义模型 ID，则使用用户指定的模型。

        Args:
            prompt: 完整的提示词

        Returns:
            生成的配文内容，失败时返回空字符串
        """
        from src.plugin_system.apis import llm_api
        from src.llm_models.utils_model import LLMRequest
        from src.config.config import model_config as maibot_model_config

        logger.info(f"[DEBUG-叙事] _call_llm() 被调用，prompt长度: {len(prompt)}")

        try:
            # 获取用户配置的自定义模型 ID
            custom_model_id = self.plugin.get_config("auto_selfie.caption_model_id", "")
            logger.info(f"[DEBUG-叙事] 配置的自定义模型ID: '{custom_model_id}'")

            # 如果用户配置了自定义模型，尝试使用
            if custom_model_id:
                available_models = llm_api.get_available_models()
                logger.info(f"[DEBUG-叙事] 可用模型列表: {list(available_models.keys()) if available_models else '无'}")
                if custom_model_id in available_models:
                    model_config = available_models[custom_model_id]
                    logger.info(f"[DEBUG-叙事] 使用用户配置的模型: {custom_model_id}")
                    logger.debug(f"使用用户配置的模型: {custom_model_id}")

                    # 调用 LLM 生成
                    success, content, reasoning, model_name = await llm_api.generate_with_model(
                        prompt=prompt,
                        model_config=model_config,
                        request_type="plugin.auto_selfie.caption_generate",
                        temperature=0.8,
                        max_tokens=9999,
                    )

                    logger.info(
                        f"[DEBUG-叙事] LLM调用结果: success={success}, content长度={len(content) if content else 0}"
                    )
                    if success and content:
                        logger.info(f"[DEBUG-叙事] LLM 生成成功，使用模型: {model_name}")
                        logger.debug(f"LLM 生成成功，使用模型: {model_name}")
                        return self._clean_caption(content)
                    else:
                        logger.warning(f"[DEBUG-叙事] LLM 生成失败: {content}")
                        logger.warning(f"LLM 生成失败: {content}")
                        return ""
                else:
                    logger.warning(f"[DEBUG-叙事] 配置的模型 '{custom_model_id}' 不存在，回退到默认 replyer 模型")
                    logger.warning(f"配置的模型 '{custom_model_id}' 不存在，回退到默认 replyer 模型")

            # 默认使用 MaiBot 的 replyer 模型（回复模型）
            # 这样配文风格与麦麦的回复保持一致
            logger.info("[DEBUG-叙事] 尝试使用 MaiBot replyer 模型")
            try:
                replyer_request = LLMRequest(
                    model_set=maibot_model_config.model_task_config.replyer,
                    request_type="plugin.auto_selfie.caption_generate",
                )
                logger.info("[DEBUG-叙事] LLMRequest 创建成功，调用 generate_response_async...")

                content, reasoning = await replyer_request.generate_response_async(
                    prompt, temperature=0.8, max_tokens=9999
                )

                logger.info(f"[DEBUG-叙事] replyer模型返回: content长度={len(content) if content else 0}")
                if content:
                    logger.info("[DEBUG-叙事] LLM 生成成功，使用 MaiBot replyer 模型")
                    logger.debug("LLM 生成成功，使用 MaiBot replyer 模型")
                    return self._clean_caption(content)
                else:
                    logger.warning("[DEBUG-叙事] replyer 模型生成失败，返回空内容")
                    logger.warning("replyer 模型生成失败，返回空内容")
                    return ""

            except Exception as e:
                import traceback

                logger.warning(f"[DEBUG-叙事] 使用 replyer 模型失败: {e}")
                logger.warning(f"[DEBUG-叙事] replyer失败堆栈: {traceback.format_exc()}")
                logger.warning(f"使用 replyer 模型失败: {e}，尝试使用 llm_api 备用方案")

                # 备用方案：使用 llm_api 的第一个可用模型
                available_models = llm_api.get_available_models()
                logger.info(
                    f"[DEBUG-叙事] 备用方案 - 可用模型: {list(available_models.keys()) if available_models else '无'}"
                )
                if available_models:
                    first_key = next(iter(available_models))
                    model_config = available_models[first_key]
                    logger.info(f"[DEBUG-叙事] 使用备用模型: {first_key}")
                    logger.debug(f"使用备用模型: {first_key}")

                    success, content, reasoning, model_name = await llm_api.generate_with_model(
                        prompt=prompt,
                        model_config=model_config,
                        request_type="plugin.auto_selfie.caption_generate",
                        temperature=0.8,
                        max_tokens=9999,
                    )

                    logger.info(
                        f"[DEBUG-叙事] 备用模型调用结果: success={success}, content长度={len(content) if content else 0}"
                    )
                    if success and content:
                        return self._clean_caption(content)
                else:
                    logger.warning("[DEBUG-叙事] 没有可用的备用模型")

                return ""

        except Exception as e:
            import traceback

            logger.error(f"[DEBUG-叙事] LLM 调用异常: {e}")
            logger.error(f"[DEBUG-叙事] 异常堆栈: {traceback.format_exc()}")
            logger.error(f"LLM 调用异常: {e}")
            return ""

    # ==================== 辅助方法 ====================

    def _clean_caption(self, raw_caption: str) -> str:
        """清理生成的配文

        去除多余的引号、空格、换行等

        Args:
            raw_caption: 原始生成的配文

        Returns:
            清理后的配文
        """
        if not raw_caption:
            return ""

        caption = raw_caption.strip()

        # 去除首尾的各种引号
        quote_chars = ['"', "'", '"', '"', """, """, "「", "」", "『", "』"]
        for char in quote_chars:
            if caption.startswith(char):
                caption = caption[1:]
            if caption.endswith(char):
                caption = caption[:-1]

        # 去除换行符
        caption = caption.replace("\n", " ").replace("\r", "")

        # 去除多余空格
        caption = re.sub(r"\s+", " ", caption).strip()

        # 去除可能的前缀（如 "配文："）
        prefixes_to_remove = ["配文：", "配文:", "Caption:", "caption:"]
        for prefix in prefixes_to_remove:
            if caption.startswith(prefix):
                caption = caption[len(prefix) :].strip()

        return caption

    def _get_fallback_caption(self, caption_type: CaptionType) -> str:
        """获取备用配文（LLM 调用失败时使用）

        Args:
            caption_type: 配文类型

        Returns:
            随机选择的备用配文
        """
        fallback_list = self.FALLBACK_CAPTIONS.get(caption_type, [])

        if not fallback_list:
            # 如果没有备用配文，返回通用配文
            return "记录生活的美好时刻"

        return random.choice(fallback_list)

    # ==================== 扩展方法 ====================

    def get_mood_emoji(self, mood: str) -> str:
        """根据情绪获取对应的 emoji

        可用于配文中增加表情

        Args:
            mood: 情绪状态

        Returns:
            对应的 emoji 字符
        """
        mood_emojis: Dict[str, List[str]] = {
            "happy": ["😊", "😄", "🥰", "✨", "💕"],
            "sad": ["😢", "😔", "🥺", "💔"],
            "tired": ["😴", "🥱", "💤", "😩"],
            "excited": ["🎉", "🤩", "💫", "⭐"],
            "neutral": ["😌", "🙂", "📷"],
            "bored": ["😑", "😶", "🫥"],
            "hungry": ["🍜", "🍕", "😋", "🤤"],
        }

        emoji_list = mood_emojis.get(mood, mood_emojis["neutral"])
        return random.choice(emoji_list)

    def add_emoji_to_caption(self, caption: str, mood: str = "neutral") -> str:
        """为配文添加情绪 emoji

        有 30% 的概率在配文末尾添加 emoji

        Args:
            caption: 原始配文
            mood: 当前情绪

        Returns:
            可能添加了 emoji 的配文
        """
        if not caption:
            return caption

        # 30% 概率添加 emoji
        if random.random() < 0.3:
            emoji = self.get_mood_emoji(mood)
            return f"{caption} {emoji}"

        return caption
