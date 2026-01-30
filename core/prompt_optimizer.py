"""提示词优化器模块

使用 MaiBot 主 LLM 将用户描述优化为专业的绘画提示词。
纯净调用，不带人设和回复风格。

v3.5.0-beta.7 更新：
- 新增 exclude_hand_actions 参数，用于自拍模式下排除手部动作描述
- 职责划分：优化器负责场景/服装/表情/光线/画质，手部动作由 _process_selfie_prompt 统一控制
"""

from typing import Tuple
from src.common.logger import get_logger
from src.plugin_system.apis import llm_api

logger = get_logger("prompt_optimizer")

# 提示词优化系统提示词（通用版本）
OPTIMIZER_SYSTEM_PROMPT = """You are a professional AI art prompt engineer. Your task is to convert user descriptions into high-quality English prompts for image generation models (Stable Diffusion, DALL-E, etc.).

## Rules:
1. Output ONLY the English prompt, no explanations or translations
2. Use comma-separated tags/phrases
3. Follow structure: subject, action/pose, scene/background, lighting, style, quality tags
4. Use weight syntax for emphasis: (keyword:1.2) for important elements
5. Keep prompts concise but descriptive (50-150 words ideal)
6. Always end with quality tags: masterpiece, best quality, high resolution

## Examples:

Input: 海边的女孩
Output: 1girl, solo, standing on beach, ocean waves, sunset sky, orange and pink clouds, warm lighting, summer dress, wind blowing hair, peaceful expression, masterpiece, best quality, high resolution

Input: 可爱的猫咪睡觉
Output: cute cat, sleeping, curled up on soft blanket, fluffy fur, closed eyes, peaceful, warm indoor lighting, cozy atmosphere, detailed fur texture, masterpiece, best quality, high resolution

Input: 赛博朋克城市
Output: cyberpunk cityscape, neon lights, futuristic buildings, flying cars, rain, reflective wet streets, holographic advertisements, purple and blue color scheme, atmospheric, cinematic lighting, masterpiece, best quality, high resolution

Now convert the following description to an English prompt:"""

# 自拍模式专用系统提示词（排除手部动作）
OPTIMIZER_SYSTEM_PROMPT_SELFIE = """You are a professional AI art prompt engineer. Your task is to convert user descriptions into high-quality English prompts for image generation models (Stable Diffusion, DALL-E, etc.).

## Rules:
1. Output ONLY the English prompt, no explanations or translations
2. Use comma-separated tags/phrases
3. Follow structure: subject, expression, scene/background, clothing, lighting, style, quality tags
4. Use weight syntax for emphasis: (keyword:1.2) for important elements
5. Keep prompts concise but descriptive (50-150 words ideal)
6. Always end with quality tags: masterpiece, best quality, high resolution

## CRITICAL - DO NOT INCLUDE:
- Any hand gestures (peace sign, thumbs up, ok sign, finger heart, waving, etc.)
- Any hand actions or poses (touching face, hand on hip, etc.)
- Any holding descriptions (holding phone, holding cup, holding anything)
- Any arm positions or movements
These will be handled by a separate system. Focus ONLY on: expression, clothing, scene, background, lighting, atmosphere.

## Examples:

Input: 自拍
Output: 1girl, solo, cute smile, looking at viewer, casual clothing, indoor scene, soft lighting, warm atmosphere, masterpiece, best quality, high resolution

Input: 在咖啡厅自拍
Output: 1girl, solo, gentle smile, looking at viewer, in a cozy cafe, coffee shop interior, warm ambient lighting, relaxed atmosphere, casual dress, masterpiece, best quality, high resolution

Input: 小乐再来张自拍看看
Output: 1girl, solo, playful expression, cute smile, looking at viewer, cozy indoor setting, soft natural lighting, comfortable atmosphere, masterpiece, best quality, high resolution

Now convert the following description to an English prompt (remember: NO hand gestures, hand actions, or holding descriptions):"""


class PromptOptimizer:
    """提示词优化器

    使用 MaiBot 主 LLM 优化用户描述为专业绘画提示词
    """

    def __init__(self, log_prefix: str = "[PromptOptimizer]"):
        self.log_prefix = log_prefix
        self._model_config = None

    def _get_model_config(self):
        """获取可用的 LLM 模型配置"""
        if self._model_config is None:
            try:
                models = llm_api.get_available_models()
                # 使用 replyer 模型（首要回复模型）
                if "replyer" in models:
                    self._model_config = models["replyer"]
                else:
                    logger.warning(f"{self.log_prefix} 没有找到 replyer 模型")
                    return None
            except Exception as e:
                logger.error(f"{self.log_prefix} 获取模型配置失败: {e}")
                return None
        return self._model_config

    async def optimize(self, user_description: str, exclude_hand_actions: bool = False) -> Tuple[bool, str]:
        """优化用户描述为专业绘画提示词

        Args:
            user_description: 用户原始描述（中文或英文）
            exclude_hand_actions: 是否排除手部动作描述（自拍模式使用）。
                                  当为 True 时，使用专用的系统提示词，
                                  明确告知 LLM 不要生成任何手势、手部动作、持物描述，
                                  这些将由 _process_selfie_prompt 统一控制。

        Returns:
            Tuple[bool, str]: (是否成功, 优化后的提示词或错误信息)
        """
        if not user_description or not user_description.strip():
            return False, "描述不能为空"

        model_config = self._get_model_config()
        if not model_config:
            # 降级：直接返回原始描述
            logger.warning(f"{self.log_prefix} 无可用模型，降级使用原始描述")
            return True, user_description

        try:
            # 根据模式选择系统提示词
            if exclude_hand_actions:
                system_prompt = OPTIMIZER_SYSTEM_PROMPT_SELFIE
                logger.debug(f"{self.log_prefix} 使用自拍模式系统提示词（排除手部动作）")
            else:
                system_prompt = OPTIMIZER_SYSTEM_PROMPT

            # 构建完整 prompt
            full_prompt = f"{system_prompt}\n\nInput: {user_description.strip()}\nOutput:"

            logger.info(f"{self.log_prefix} 开始优化提示词: {user_description[:50]}...")

            # 调用 LLM（不传递 temperature 和 max_tokens，使用模型默认值）
            success, response, reasoning, model_name = await llm_api.generate_with_model(
                prompt=full_prompt,
                model_config=model_config,
                request_type="plugin.prompt_optimize",
            )

            if success and response:
                # 清理响应（移除可能的前缀/后缀）
                optimized = self._clean_response(response)
                logger.info(f"{self.log_prefix} 优化成功 (模型: {model_name}): {optimized[:80]}...")
                return True, optimized
            else:
                logger.warning(f"{self.log_prefix} LLM 返回空响应，降级使用原始描述: {user_description[:50]}...")
                return True, user_description

        except Exception as e:
            logger.error(f"{self.log_prefix} 优化失败: {e}，使用原始描述: {user_description[:50]}...")
            # 降级：返回原始描述
            return True, user_description

    def _clean_response(self, response: str) -> str:
        """清理 LLM 响应

        移除可能的前缀、后缀、引号等
        """
        result = response.strip()

        # 移除可能的 "Output:" 前缀
        prefixes_to_remove = ["Output:", "output:", "Prompt:", "prompt:"]
        for prefix in prefixes_to_remove:
            if result.startswith(prefix):
                result = result[len(prefix) :].strip()

        # 移除首尾引号
        if (result.startswith('"') and result.endswith('"')) or (result.startswith("'") and result.endswith("'")):
            result = result[1:-1]

        # 移除多余换行
        result = " ".join(result.split())

        return result


# 全局优化器实例
_optimizer_instance = None


def get_optimizer(log_prefix: str = "[PromptOptimizer]") -> PromptOptimizer:
    """获取提示词优化器实例（单例）"""
    global _optimizer_instance
    if _optimizer_instance is None:
        _optimizer_instance = PromptOptimizer(log_prefix)
    return _optimizer_instance


async def optimize_prompt(
    user_description: str, log_prefix: str = "[PromptOptimizer]", exclude_hand_actions: bool = False
) -> Tuple[bool, str]:
    """便捷函数：优化提示词

    Args:
        user_description: 用户原始描述
        log_prefix: 日志前缀
        exclude_hand_actions: 是否排除手部动作描述（自拍模式使用）。
                              当为 True 时，LLM 不会生成任何手势、手部动作、持物描述，
                              这些将由 _process_selfie_prompt 统一控制。

    Returns:
        Tuple[bool, str]: (是否成功, 优化后的提示词)
    """
    optimizer = get_optimizer(log_prefix)
    return await optimizer.optimize(user_description, exclude_hand_actions=exclude_hand_actions)
