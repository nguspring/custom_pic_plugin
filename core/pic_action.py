import asyncio
import traceback
import base64
import os
from typing import List, Tuple, Type, Optional, Dict, Any

import aiohttp  # type: ignore[import-not-found]

from src.plugin_system.base.base_action import BaseAction
from src.plugin_system.base.component_types import ActionActivationType, ChatMode
from src.common.logger import get_logger

from .api_clients import get_client_class
from .image_utils import ImageProcessor
from .cache_manager import CacheManager
from .size_utils import validate_image_size, get_image_size
from .runtime_state import runtime_state
from .prompt_optimizer import optimize_prompt
from .image_search_adapter import ImageSearchAdapter

logger = get_logger("pic_action")

class CustomPicAction(BaseAction):
    """统一的图片生成动作，智能检测文生图或图生图"""

    # 激活设置
    activation_type = ActionActivationType.ALWAYS  # 默认激活类型
    focus_activation_type = ActionActivationType.ALWAYS  # Focus模式使用LLM判定，精确理解需求
    normal_activation_type = ActionActivationType.KEYWORD  # Normal模式使用关键词激活，快速响应
    mode_enable = ChatMode.ALL
    parallel_action = True

    # 动作基本信息
    action_name = "draw_picture"
    action_description = (
        "智能图片生成：根据描述生成图片（文生图）或基于现有图片进行修改（图生图）。"
        "自动检测用户是否提供了输入图片来决定使用文生图还是图生图模式。"
        "支持多种API格式：OpenAI、豆包、Gemini、硅基流动、魔搭社区、砂糖云(NovelAI)、ComfyUI、梦羽AI等。"
    )

    # 关键词设置（用于Normal模式）
    activation_keywords = [
        # 文生图关键词
        "画", "绘制", "生成图片", "画图", "draw", "paint", "图片生成", "创作",
        # 图生图关键词
        "图生图", "修改图片", "基于这张图", "img2img", "重画", "改图", "图片修改",
        "改成", "换成", "变成", "转换成", "风格", "画风", "改风格", "换风格",
        "这张图", "这个图", "图片风格", "改画风", "重新画", "再画", "重做",
        # 自拍关键词
        "自拍", "selfie", "拍照", "对镜自拍", "镜子自拍", "照镜子"
    ]

    # LLM判定提示词（用于Focus模式）
    ALWAYS_prompt = """
判定是否需要使用图片生成动作的条件：

**文生图场景：**
1. 用户明确@你的名字并要求画图、生成图片或创作图像
2. 用户描述了想要看到的画面或场景
3. 对话中提到需要视觉化展示某些概念
4. 用户想要创意图片或艺术作品
5. 你想要通过画图来制作表情包表达情绪

**图生图场景：**
1. 用户发送了图片并@你的名字要求基于该图片进行修改或重新生成
2. 用户明确@你的名字要求并提到"图生图"、"修改图片"、"基于这张图"等关键词
3. 用户想要改变现有图片的风格、颜色、内容等
4. 用户要求在现有图片基础上添加或删除元素

**自拍场景：**
1. 用户明确要求你进行自拍、拍照等
2. 用户提到"自拍"、"selfie"、"照镜子"、"对镜自拍"等关键词
3. 用户想要看到你的照片或形象

**绝对不要使用的情况：**
1. 纯文字聊天和问答
2. 只是提到"图片"、"画"等词但不是要求生成
3. 谈论已存在的图片或照片（仅讨论不修改）
4. 技术讨论中提到绘图概念但无生成需求
5. 用户明确表示不需要图片时
6. 刚刚成功生成过图片，避免频繁请求
"""

    keyword_case_sensitive = False

    # 动作参数定义（简化版，提示词优化由独立模块处理）
    action_parameters = {
        "description": "从用户消息中提取的图片描述文本（例如：用户说'画一只小猫'，则填写'一只小猫'）。必填参数。",
        "model_id": """要使用的模型ID（如model1、model2、model3等）。
        重要：需要从用户消息中提取模型ID！  
        支持的自然语言表达方式：
        - '用model3画一只猫' → 提取 'model3'
        - 'model2生成图片' → 提取 'model2'
        - '使用模型1发张自拍' → 提取 'model1'
        - '用模型1'、'模型2画'、'模型3生成'等   
        如果用户没有指定模型，则留空或填null（将使用默认模型）""",
        "strength": "图生图强度，0.1-1.0之间，值越高变化越大（仅图生图时使用，可选，默认0.7）",
        "size": "图片尺寸，如512x512、1024x1024等（可选，不指定则使用模型默认尺寸）",
        "selfie_mode": "是否启用自拍模式（true/false，可选，默认false）。启用后会自动添加自拍场景和手部动作",
        "selfie_style": "自拍风格，可选值：standard（标准自拍，适用于户外或无镜子场景），mirror（对镜自拍，适用于有镜子的室内场景）。仅在selfie_mode=true时生效，可选，默认standard",
        "free_hand_action": "自由手部动作描述（英文）。如果指定此参数，将使用此动作而不是随机生成。仅在selfie_mode=true时生效，可选"
    }

    # 动作使用场景
    action_require = [
        "当用户要求生成或修改图片时使用，不要频率太高",
        "自动检测是否有输入图片来决定文生图或图生图模式",
        "重点：不要连续发，如果你在前10句内已经发送过[图片]或者[表情包]或记录出现过类似描述的[图片]，就不要选择此动作",
        # 新增说明
        "【重要】模型指定规则：如果用户明确提到特定模型，必须在model_id参数中填写！",
        "支持的表达方式（必须提取）：",
        "  - '用model3画' → model_id='model3'",
        "  - 'model2生成' → model_id='model2'",
        "  - '使用模型1' → model_id='model1'",
        "  - '用模型1发个自拍' → model_id='model1', selfie_mode=true",
        "  - 类似'用...画'、'...生成'、'...发'等表达都要提取模型ID",
        "  - 中文'模型1'、'模型2'、'模型3'对应 model1、model2、model3",
        "如果用户没有指定任何模型，则model_id留空（将使用默认模型default_model）"
    ]
    associated_types = ["text", "image"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_processor = ImageProcessor(self)
        self.cache_manager = CacheManager(self)
        self._api_clients = {}  # 缓存不同格式的API客户端

    def _get_api_client(self, api_format: str):
        """获取指定格式的API客户端（带缓存）"""
        if api_format not in self._api_clients:
            client_class = get_client_class(api_format)
            self._api_clients[api_format] = client_class(self)
        return self._api_clients[api_format]

    async def execute(self) -> Tuple[bool, Optional[str]]:
        """执行统一图片生成动作"""
        logger.info(f"{self.log_prefix} 执行统一图片生成动作")

        # 检查是否是 /dr 命令消息，如果是则跳过（由 Command 组件处理）
        if self.action_message and self.action_message.processed_plain_text:
            message_text = self.action_message.processed_plain_text.strip()
            # 修正：将第二个 if 放在第一个 if 内部，或者确保 message_text 总是被定义
            if message_text.startswith("/dr ") or message_text == "/dr":
                logger.info(f"{self.log_prefix} 检测到 /dr 命令，跳过 Action 处理（由 Command 组件处理）")
                return False, "跳过 /dr 命令"

        # 检查插件是否在当前聊天流启用
        global_enabled = bool(self.get_config("plugin.enabled", True))
        if not runtime_state.is_plugin_enabled(self.chat_id, global_enabled):
            logger.info(f"{self.log_prefix} 插件在当前聊天流已禁用")
            # 修正：return 需要缩进在 if 内部
            return False, "插件已禁用"

        # 获取参数
        description = (self.action_data.get("description") or "").strip()
        model_id = (self.action_data.get("model_id") or "").strip()
        strength = self.action_data.get("strength", 0.7)
        size = (self.action_data.get("size") or "").strip()
        selfie_mode = self.action_data.get("selfie_mode", False)
        selfie_style = (self.action_data.get("selfie_style") or "standard").strip().lower()
        free_hand_action = (self.action_data.get("free_hand_action") or "").strip()


        # 如果没有指定模型，使用运行时状态的默认模型
        if not model_id:
            global_default = str(self.get_config("generation.default_model", "model1"))
            model_id = runtime_state.get_action_default_model(self.chat_id, global_default)

        # 检查模型是否在当前聊天流启用
        if not runtime_state.is_model_enabled(self.chat_id, model_id):
            logger.warning(f"{self.log_prefix} 模型 {model_id} 在当前聊天流已禁用")
            # 修正：缩进
            await self.send_text(f"模型 {model_id} 当前不可用")
            return False, f"模型 {model_id} 已禁用"

        # 参数验证和后备提取
        if not description:
            # 尝试从action_message中提取描述
            extracted_description = self._extract_description_from_message()
            if extracted_description:
                description = extracted_description
                logger.info(f"{self.log_prefix} 从消息中提取到图片描述: {description}")
            else:
                logger.warning(f"{self.log_prefix} 图片描述为空，无法生成图片。")
                # 修正：缩进
                await self.send_text("你需要告诉我想要画什么样的图片哦~ 比如说'画一只可爱的小猫'")
                return False, "图片描述为空"

        # 清理和验证描述
        if len(description) > 1000:
            description = description[:1000]
            logger.info(f"{self.log_prefix} 图片描述过长，已截断至1000字符")

        # ============================================================
        # 【智能参考搜索】新增代码块开始
        # ============================================================
        
        # 1. 检查配置里有没有开启这个功能
        ref_search_enabled = self.get_config("search_reference.enabled", False)
        
        # 2. 只有开启了功能，且不是自拍模式时才执行（避免冲突）
        if ref_search_enabled and not selfie_mode:
            logger.info(f"{self.log_prefix} 触发智能参考搜索: {description}")
            
            try:
                # 3. 调用图片搜索适配器去搜图
                image_url = await ImageSearchAdapter.search(description, max_results=3)
                
                # 如果搜到了图片链接
                if image_url:
                    features = ""
                    
                    # 4. 检查用户是否配置了自定义视觉模型
                    v_api_key = str(self.get_config("search_reference.vision_api_key", "")).strip()
                    
                    if v_api_key:
                        # 用户配置了自定义模型，使用 VisionAnalyzer
                        from .vision_analyzer import VisionAnalyzer
                        
                        v_base_url = str(self.get_config("search_reference.vision_base_url", "https://api.openai.com/v1"))
                        v_model = str(self.get_config("search_reference.vision_model", "gpt-4o"))
                        
                        logger.info(f"{self.log_prefix} 使用用户自定义视觉模型: {v_model}")
                        analyzer = VisionAnalyzer(v_base_url, v_api_key, v_model)
                        features = await analyzer.analyze_image(image_url)
                    else:
                        # 默认使用 MaiBot 的 vlm 模型（视觉语言模型）
                        logger.info(f"{self.log_prefix} 使用 MaiBot 默认 vlm 模型")
                        try:
                            from src.llm_models.utils_model import LLMRequest
                            from src.config.config import model_config as maibot_model_config
                            
                            # 下载图片并转为 Base64
                            img_base64_data = None
                            async with aiohttp.ClientSession() as session:
                                async with session.get(image_url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                                    if resp.status == 200:
                                        image_bytes = await resp.read()
                                        img_base64_data = base64.b64encode(image_bytes).decode('utf-8')
                                    else:
                                        logger.warning(f"{self.log_prefix} 下载参考图片失败: {resp.status}")
                            
                            if img_base64_data:
                                # 使用 MaiBot 的 vlm 模型分析图片
                                vlm_request = LLMRequest(
                                    model_set=maibot_model_config.model_task_config.vlm,
                                    request_type="plugin.search_reference.vision_analyze"
                                )
                                
                                # 构造视觉分析的 prompt
                                vision_prompt = (
                                    "请详细分析这张图片中的角色视觉特征。"
                                    "提取关键特征并转化为 Stable Diffusion 格式的英文提示词（Tag）。"
                                    "包括但不限于：发色、瞳色、发型、服装、配饰、姿势、背景风格等。"
                                    "只需返回提示词，不要包含任何解释性文字。"
                                )
                                
                                # 调用 vlm 的 generate_response_for_image 方法
                                # 根据图片 URL 判断格式，默认为 jpeg
                                img_format = "jpeg"
                                if image_url.lower().endswith(".png"):
                                    img_format = "png"
                                elif image_url.lower().endswith(".webp"):
                                    img_format = "webp"
                                elif image_url.lower().endswith(".gif"):
                                    img_format = "gif"
                                
                                result = await vlm_request.generate_response_for_image(
                                    prompt=vision_prompt,
                                    image_base64=img_base64_data,
                                    image_format=img_format
                                )
                                
                                # 解析返回结果
                                if result and len(result) >= 1:
                                    features = result[0] if result[0] else ""
                                    if features:
                                        logger.info(f"{self.log_prefix} VLM 分析成功: {features[:100]}...")
                                    
                        except Exception as vlm_e:
                            logger.warning(f"{self.log_prefix} MaiBot vlm 模型分析失败: {vlm_e}")
                            features = ""
                    
                    # 5. 如果分析成功，就把特征拼接到用户的描述里
                    if features:
                        # 拼接格式：原描述, (提取的特征:1.3)
                        # 1.3 是权重，表示让模型更重视这些特征
                        description = f"{description}, ({features}:1.3)"
            
            except Exception as e:
                # 如果中间出错了（比如网络断了），记录日志，但不要让整个程序崩掉
                logger.error(f"{self.log_prefix} 智能参考搜索出错: {e}", exc_info=True)
                # 即使出错，也继续往下跑，让用户至少能拿到一张普通的图

        # ============================================================
        # 【智能参考搜索】新增代码块结束
        # ============================================================

        # 提示词优化
        optimizer_enabled = self.get_config("prompt_optimizer.enabled", True)
        if optimizer_enabled:
            logger.info(f"{self.log_prefix} 开始优化提示词: {description}")#显示所有提示词
            success, optimized_prompt = await optimize_prompt(description, self.log_prefix)
            # 修正：if success 需要缩进在 optimizer_enabled if 内部
            if success:
                logger.info(f"{self.log_prefix} 提示词优化完成: {optimized_prompt}")#显示所有提示词
                description = optimized_prompt
            else:
                logger.warning(f"{self.log_prefix} 提示词优化失败，使用原始描述: {description}")#显示所有提示词

        # 验证strength参数
        try:
            strength = float(strength)
            # 修正：if 检查需要缩进在 try 内部
            if not (0.1 <= strength <= 1.0):
                strength = 0.7
        except (ValueError, TypeError):
            strength = 0.7

        # 👇【新增修复代码】在这里初始化变量，给一个空字符串作为默认值
        selfie_negative_prompt = ""

        # 处理自拍模式
        if selfie_mode:
            # 检查自拍功能是否启用
            # 修正：缩进
            selfie_enabled = self.get_config("selfie.enabled", True)
            if not selfie_enabled:
                # 修正：缩进
                await self.send_text("自拍功能暂未启用~")
                return False, "自拍功能未启用"

            logger.info(f"{self.log_prefix} 启用自拍模式，风格: {selfie_style}")
            description = self._process_selfie_prompt(description, selfie_style, free_hand_action, model_id)
            logger.info(f"{self.log_prefix} 自拍模式处理后的提示词: {description}") # 显示所有提示词

            # 👇 读取自拍专用负面提示词（从配置读取基础负面词） 👇
            selfie_negative_prompt = str(self.get_config("selfie.negative_prompt", "")).strip()
            
            # 👇 【修复双手问题】standard模式专用负面提示词，防止生成两只手 👇
            if selfie_style == "standard":
                # 定义防止双手/双臂的负面提示词
                standard_anti_dual_hands = (
                    # 防止双手持机
                    "two phones, camera in both hands, holding phone with both hands, "
                    "extra hands, extra arms, 3 hands, 4 hands, multiple hands, "
                    "both hands holding phone, two hands on phone, "
                    # 防止手机出现在画面中
                    "phone in frame, visible phone in hand, phone screen visible, "
                    "floating phone, phone reflection, smartphone visible, "
                    # 防止两只手都出现
                    "both hands visible, two hands making gesture, "
                    "holding device with two hands, dual arm selfie, "
                    "symmetrical hands, mirrored hands, "
                    # 防止持机手边缘可见（新增）
                    "hand at edge of frame, hand entering frame from side, "
                    "partial hand visible at edge, hand reaching into frame, "
                    "fingers at edge of frame, palm at edge, "
                    "hand extending from outside frame, arm entering frame, "
                    "hand peeking from edge, visible phone holding hand"
                )
                # 合并：用户配置的负面词 + standard模式专用防双手词
                if selfie_negative_prompt:
                    selfie_negative_prompt = f"{selfie_negative_prompt}, {standard_anti_dual_hands}"
                else:
                    selfie_negative_prompt = standard_anti_dual_hands
                logger.info(f"{self.log_prefix} 已应用standard模式防双手负面提示词")
            # 👆 【修复双手问题】结束 👆

            # 检查是否配置了参考图片
            reference_image = self._get_selfie_reference_image()
            if reference_image:
                # 检查模型是否支持图生图
                model_config = self._get_model_config(model_id)
                if model_config and model_config.get("support_img2img", True):
                    logger.info(f"{self.log_prefix} 使用自拍参考图片进行图生图")
                    return await self._execute_unified_generation(description, model_id, size, strength or 0.6, reference_image, selfie_negative_prompt) #修改：增加selfie_negative_prompt
                else:
                    logger.warning(f"{self.log_prefix} 模型 {model_id} 不支持图生图，自拍回退为文生图模式")
            # 无参考图或模型不支持，继续使用文生图（向下执行）

        # **智能检测：判断是文生图还是图生图**
        input_image_base64 = await self.image_processor.get_recent_image()
        is_img2img_mode = input_image_base64 is not None

        if is_img2img_mode:
            # 检查指定模型是否支持图生图
            model_config = self._get_model_config(model_id)
            if model_config and not model_config.get("support_img2img", True):
                logger.warning(f"{self.log_prefix} 模型 {model_id} 不支持图生图，转为文生图模式")
                await self.send_text(f"当前模型 {model_id} 不支持图生图功能，将为您生成新图片")
                return await self._execute_unified_generation(description, model_id, size, None, None)

            logger.info(f"{self.log_prefix} 检测到输入图片，使用图生图模式")
            return await self._execute_unified_generation(description, model_id, size, strength, input_image_base64)
        else:
            logger.info(f"{self.log_prefix} 未检测到输入图片，使用文生图模式")
            return await self._execute_unified_generation(description, model_id, size, None, None, selfie_negative_prompt) #修改：增加selfie_negative_prompt

    # 👇 新增参数 extra_negative_prompt: str = None
    async def _execute_unified_generation(self, description: str, model_id: str, size: str, strength: Optional[float] = None, input_image_base64: Optional[str] = None, extra_negative_prompt: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """统一的图片生成执行方法"""
        # 调用内部方法生成图片
        success, image_data, error_msg = await self._generate_image_internal(
            description=description,
            model_id=model_id,
            size=size,
            strength=strength,
            input_image_base64=input_image_base64,
            extra_negative_prompt=extra_negative_prompt
        )
        
        if not success or not image_data:
            is_img2img = input_image_base64 is not None
            mode_text = "图生图" if is_img2img else "文生图"
            await self.send_text(f"哎呀，{mode_text}时遇到问题：{error_msg}")
            return False, f"{mode_text}失败: {error_msg}"
        
        # 发送图片
        is_img2img = input_image_base64 is not None
        enable_debug = self.get_config("components.enable_debug_info", False)
        
        send_success = await self.send_image(image_data)
        if send_success:
            mode_text = "图生图" if is_img2img else "文生图"
            if enable_debug:
                await self.send_text(f"{mode_text}完成！")
            # 缓存成功的结果
            model_config = self._get_model_config(model_id)
            model_name = model_config.get("model", "default-model") if model_config else "default-model"
            image_size = size or model_config.get("default_size", "1024x1024") if model_config else "1024x1024"
            self.cache_manager.cache_result(description, model_name, image_size, strength, is_img2img, image_data)
            # 安排自动撤回（如果该模型启用）
            await self._schedule_auto_recall_for_recent_message(model_config)
            return True, f"{mode_text}已成功生成并发送"
        else:
            await self.send_text("图片已处理完成，但发送失败了")
            return False, "图片发送失败"

    async def _generate_image_only(self, description: str, model_id: str, size: str, strength: Optional[float] = None, input_image_base64: Optional[str] = None, extra_negative_prompt: Optional[str] = None) -> Optional[str]:
        """仅生成图片，返回 base64 编码，不发送
        
        用于自动自拍任务的"生成一次，发送多次"模式
        
        Args:
            description: 图片描述
            model_id: 模型ID
            size: 图片尺寸
            strength: 图生图强度
            input_image_base64: 输入图片的 base64 编码
            extra_negative_prompt: 额外的负面提示词
            
        Returns:
            Optional[str]: 图片的 base64 编码，失败返回 None
        """
        success, image_data, error_msg = await self._generate_image_internal(
            description=description,
            model_id=model_id,
            size=size,
            strength=strength,
            input_image_base64=input_image_base64,
            extra_negative_prompt=extra_negative_prompt
        )
        
        if success and image_data:
            return image_data
        else:
            logger.warning(f"{self.log_prefix} 图片生成失败: {error_msg}")
            return None

    async def _generate_image_internal(self, description: str, model_id: str, size: str, strength: Optional[float] = None, input_image_base64: Optional[str] = None, extra_negative_prompt: Optional[str] = None) -> Tuple[bool, Optional[str], str]:
        """内部图片生成方法，返回 (成功, 图片base64, 错误信息)
        
        注意：此方法只生成图片，不发送。发送由调用者负责。
        """

        # 获取模型配置
        model_config = self._get_model_config(model_id)
        if not model_config:
            error_msg = f"指定的模型 '{model_id}' 不存在或配置无效"
            logger.error(f"{self.log_prefix} 模型配置获取失败: {model_id}")
            return False, None, error_msg

        # 配置验证
        http_base_url = model_config.get("base_url")
        http_api_key = model_config.get("api_key")
        if not (http_base_url and http_api_key):
            error_msg = "HTTP配置不完整（缺少base_url或api_key）"
            logger.error(f"{self.log_prefix} HTTP调用配置缺失: base_url 或 api_key.")
            return False, None, error_msg

        # API密钥验证
        if "YOUR_API_KEY_HERE" in http_api_key or "xxxxxxxxxxxxxx" in http_api_key:
            error_msg = "API密钥未配置"
            logger.error(f"{self.log_prefix} API密钥未配置")
            return False, None, error_msg

        # 获取模型配置参数
        model_name = model_config.get("model", "default-model")
        api_format = model_config.get("format", "openai")

        # 合并负面提示词
        if extra_negative_prompt:
            # 复制一份配置，避免修改原始配置影响后续调用
            model_config = dict(model_config)
            original_neg = model_config.get("negative_prompt_add", "")
            # 合并：原有负面词 + 自拍专用负面词
            combined_neg = f"{original_neg}, {extra_negative_prompt}".strip(", ")
            model_config["negative_prompt_add"] = combined_neg
            logger.info(f"{self.log_prefix} 已应用自拍专用负面提示词: {extra_negative_prompt[:50]}...")

        # 使用统一的尺寸处理逻辑
        image_size, llm_original_size = get_image_size(model_config, size, self.log_prefix)

        # 验证图片尺寸格式
        if not self._validate_image_size(image_size):
            logger.warning(f"{self.log_prefix} 无效的图片尺寸: {image_size}，使用模型默认值")
            image_size = model_config.get("default_size", "1024x1024")

        # 检查缓存
        is_img2img = input_image_base64 is not None
        cached_result = self.cache_manager.get_cached_result(description, model_name, image_size, strength, is_img2img)

        if cached_result:
            logger.info(f"{self.log_prefix} 使用缓存的图片结果")
            return True, cached_result, ""

        try:
            # 对于 Gemini/Zai 格式，将原始 LLM 尺寸添加到 model_config 中
            if api_format in ("gemini", "zai") and llm_original_size:
                model_config = dict(model_config)  # 创建副本避免修改原配置
                model_config["_llm_original_size"] = llm_original_size

            # 获取重试次数配置
            max_retries = self.get_config("components.max_retries", 2)

            # 获取对应格式的API客户端并调用
            api_client = self._get_api_client(api_format)
            success, result = await api_client.generate_image(
                prompt=description,
                model_config=model_config,
                size=image_size,
                strength=strength,
                input_image_base64=input_image_base64,
                max_retries=max_retries
            )
        except Exception as e:
            logger.error(f"{self.log_prefix} 异步请求执行失败: {e!r}", exc_info=True)
            traceback.print_exc()
            return False, None, f"图片生成服务遇到意外问题: {str(e)[:100]}"

        if success and result:
            final_image_data = self.image_processor.process_api_response(result)

            if final_image_data:
                if final_image_data.startswith(("iVBORw", "/9j/", "UklGR", "R0lGOD")):  # Base64
                    # 缓存成功的结果
                    self.cache_manager.cache_result(description, model_name, image_size, strength, is_img2img, final_image_data)
                    return True, final_image_data, ""
                else:  # URL - 需要下载转换为 base64
                    try:
                        encode_success, encode_result = await asyncio.to_thread(
                            self.image_processor.download_and_encode_base64, final_image_data
                        )
                        if encode_success and encode_result:
                            # 缓存成功结果
                            self.cache_manager.cache_result(description, model_name, image_size, strength, is_img2img, encode_result)
                            return True, encode_result, ""
                        else:
                            return False, None, f"图片处理失败: {encode_result}"
                    except Exception as e:
                        logger.error(f"{self.log_prefix} 图片下载编码失败: {e!r}")
                        return False, None, "图片下载失败"
            else:
                return False, None, "API返回数据格式错误"
        else:
            is_img2img = input_image_base64 is not None
            mode_text = "图生图" if is_img2img else "文生图"
            return False, None, f"{mode_text}失败: {result}"

    def _get_model_config(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """获取指定模型的配置，支持热重载"""
        # 如果没有指定模型ID，使用默认模型
        if not model_id:
            model_id = str(self.get_config("generation.default_model", "model1"))

        # 构建模型配置的路径
        model_config_path = f"models.{model_id}"
        model_config = self.get_config(model_config_path)

        if not model_config:
            logger.warning(f"{self.log_prefix} 模型 {model_id} 配置不存在，尝试使用默认模型")
            # 尝试获取默认模型
            default_model_id = self.get_config("generation.default_model", "model1")
            if default_model_id != model_id:
                model_config = self.get_config(f"models.{default_model_id}")

        if isinstance(model_config, dict):
            return model_config
        return {}

    def _validate_image_size(self, size: str) -> bool:
        """验证图片尺寸格式是否正确（委托给size_utils）"""
        return validate_image_size(size)

    def _process_selfie_prompt(self, description: str, selfie_style: str, free_hand_action: str, model_id: str) -> str:
        """处理自拍模式的提示词生成"""
        import random
        import re  # 导入正则库，用于清理冲突词

        # 1. 添加强制主体设置
        forced_subject = "(1girl:1.4), (solo:1.3)"

        # 2. 从独立的selfie配置中获取Bot的默认形象特征
        bot_appearance = str(self.get_config("selfie.prompt_prefix", "")).strip()

        # 3. 定义自拍风格特定的场景设置（通用版：适用于真实风格和二次元风格）
        if selfie_style == "mirror":
            # 对镜自拍：强调倒影、手机在手、室内场景
            default_mirror = "mirror selfie, reflection in mirror, holding phone in hand, phone visible, arm slightly bent, looking at mirror, indoor scene, soft lighting, high quality"
            selfie_scene = str(self.get_config("selfie.scene_mirror", default_mirror))
        else:
            # 前置自拍：强调场景和视角（手部约束由 hand_action 处理）
            # 精简版：移除与 hand_action 重复的手部描述
            default_standard = (
                "selfie, front camera view, POV selfie, "
                "(front facing selfie camera angle:1.3), "
                "looking at camera, slight high angle selfie, "
                "upper body shot, cowboy shot, "
                "(centered composition:1.2)"
            )
            selfie_scene = str(self.get_config("selfie.scene_standard", default_standard))

        # 4. 智能手部动作库（比原版更多的动作！）
        hand_actions = [
            # --- 经典单手手势 ---
            "peace sign, v sign",                     # 剪刀手（自拍最经典）
            "thumbs up, positive gesture",            # 竖大拇指
            "thumbs down, negative gesture",          # 倒大拇指
            "ok sign, hand gesture",                  # OK手势
            "rock on sign, heavy metal gesture",      # 摇滚手势（金属礼）
            "shaka sign, hang loose",                 # 悬挂手势（小拇指和大拇指伸出）
            "call me hand gesture",                   # "打电话"手势（六字手势）
            "pointing at camera lens, engaging",      # 手指指镜头（互动感强）
            "fist pump, excited",                     # 单手挥拳（兴奋）
            "saluting with one hand",                 # 单手敬礼
            "clenched fist, fighting spirit",         # 握紧拳头（元气）
            "crossing fingers, wishing luck",          # 单手交叉手指（祈祷好运）
            "showing palm, stop gesture",             # 手掌摊开（停止/五指张开）

            # --- 面部与头部互动（特写感） ---
            "touching own cheek gently",              # 轻轻摸自己的脸
            "leaning chin on hand, cute",             # 托腮（需侧身或对镜）
            "hand near chin, thinking pose",          # 手靠近下巴（思考）
            "covering mouth with hand, shy giggle",   # 手遮嘴笑（害羞）
            "finger on lips, shushing",               # 食指按唇（嘘）
            "hand covering one eye, peeking",         # 遮住一只眼偷看
            "playing with hair, messy look",          # 玩弄头发
            "tucking hair behind ear",                # 把头发别在耳后
            "fixing fringe, adjusting hair",           # 整理刘海
            "hand on forehead, dramatic",             # 手扶额头（无奈/戏剧感）
            "scratching head, confused",              # 挠头（困惑）
            "pulling collar, flustered",              # 拉衣领（热/慌乱）
            "touching neck, elegant",                 # 摸脖子（优雅）
            "supporting jaw with hand",               # 手撑下巴（特写）

            # --- 身体姿态与时尚 ---
            "hand on hip, confident",                 # 单手叉腰（最显瘦姿势）
            "hand akimbo, sassy",                     # 叉腰（傲娇）
            "hand behind head, relaxed cool",          # 手放在脑后（放松/对镜）
            "hand resting on shoulder",               # 手搭在肩膀上（防御/可爱）
            "adjusting sleeve, detail",               # 整理袖子
            "fixing collar, neat",                    # 整理衣领
            "adjusting earring",                      # 调整耳环
            "wearing sunglasses on face",             # 戴上墨镜
            "holding sunglasses, looking down",       # 手拿墨镜
            "hand touching necklace",                 # 摸项链
            "hand in pocket, casual",                 # 另一只手插兜（酷）
            "resting arm on leg",                     # 手臂搭在腿上（坐姿自拍）
            "hand on wall, leaning pose",             # 手撑墙（对镜/侧身）
            "hand on table, relaxing",                # 手放在桌上（咖啡店风格）

            # --- 甜美与可爱 ---
            "finger heart, cute pose",                # 单手指比心（韩系）
            "blowing kiss, romantic",                 # 飞吻
            "cat paw gesture, playful",               # 猫爪手势
            "bunny ears with fingers",                # 手指比兔耳
            "holding invisible ball",                 # 抱着隐形球
            "winking with hand near face",            # 手靠近脸部眨眼
            "pinky promise",                          # 拉钩手势
            "making a heart shape with one arm",      # 单臂弯曲成心形
            "claw gesture, cute monster",             # 爪子手势
            "framing face with hand",                 # 手做框住脸

            # --- 单手持物互动（小物件） ---
            "holding coffee cup, steam rising",       # 拿着咖啡杯
            "drinking from a straw",                  # 喝饮料（吸管）
            "holding a milk tea bubble tea",          # 拿着奶茶
            "holding a can of soda",                  # 拿着汽水罐
            "holding a lollipop, colorful",           # 拿着棒棒糖
            "eating ice cream, happy",                # 吃冰淇淋
            "holding a flower, smelling it",          # 拿着花闻
            "holding a bouquet of flowers",            # 抱着一束花
            "holding a plush toy",                    # 拿着毛绒公仔
            "holding a cute mascot doll",              # 拿着玩偶
            "holding a pen, thinking",                # 拿着笔思考
            "holding a book, reading",                # 拿着书（展示封面）
            "holding a fashion magazine",             # 拿着时尚杂志
            "holding a microphone, singing",          # 拿着麦克风
            "holding a game controller",              # 拿着手柄（需另一只手拿设备自拍）
            "holding a game console (Switch)",        # 拿着游戏机
            "holding a musical instrument (ukulele)", # 拿着尤克里里
            "holding a camera strap",                 # 拿着相机背带
            "holding a fan",                          # 拿着扇子
            "wearing a watch on wrist",               # 亮出手表（特写）
            "wearing a bracelet",                     # 亮出手链

            # --- 指向与引导 ---
            "pointing at viewer, engaging",           # 指向观众
            "pointing up, eureka",                    # 指向上方
            "pointing sideways, look here",           # 指向旁边
            "beckoning with finger",                  # 勾手指（过来）
            "thumbs pointing behind",                 # 大拇指指向身后
            "waving hand, greeting",                  # 挥手打招呼

            # --- 特殊视角与对镜自拍特有 ---
            "hand reaching out to camera",            # 手伸向镜头（透视感）
            "hand touching the camera lens",          # 手摸镜头（模糊/接触感）
            "hand resting on chin, close-up",         # 托腮大特写
            "hand covering part of face",             # 手遮住部分脸（构图感）
            "hand forming a frame",                   # 手做取景框
            "peace sign under chin",                  # 剪刀手在下巴
            "showing fingernails, manicure",          # 展示指甲（美甲特写）
            "palm resting on cheek, cute",            # 手掌贴脸
            "fist under chin",                        # 拳头托下巴
            "elbow on table, hand supporting head",   # 肘部撑桌手托头
        ]


        # 5. 选择手部动作
        if free_hand_action:
            hand_action = free_hand_action
        else:
            hand_action = random.choice(hand_actions)
        
        # 👇 修复双手问题：在standard模式下，明确描述自由手动作，强调持机手完全不可见 👇
        if selfie_style == "standard":
            # 构建自由手动作描述（明确是"可见的那只手"在做动作）
            hand_action = (
                f"(visible free hand {hand_action}:1.4), "  # 自由手在做的动作
                "(only one hand visible in frame:1.6), "     # 画面中只能看到一只手（权重提高）
                "(single hand gesture:1.4), "                # 单手手势
                "(other hand completely outside frame:1.7), "  # 另一只手完全在画面外（权重提高）
                "(phone holding hand not visible at all:1.6), "  # 持机手完全不可见（新增）
                "(arm holding device fully cropped:1.5), "  # 持机手臂完全被裁切（新增）
                "(no part of phone hand visible:1.5), "     # 持机手任何部分都不可见（新增）
                "(selfie POV with one arm extended outside:1.4), "   # 自拍视角，手臂伸出画面外
                "(front camera perspective:1.2), "           # 前置摄像头视角
                "(subject centered in frame:1.3)"            # 人物居中构图（新增）
            )
        # 👆 修复双手问题结束 👆

        # 6. 组装完整提示词
        prompt_parts = [forced_subject]

        if bot_appearance:
            prompt_parts.append(bot_appearance)

        prompt_parts.extend([
            hand_action,
            selfie_scene,
            description  # 这里包含了优化器加的 "holding a smartphone"
        ])

        # 7. 合并
        final_prompt = ", ".join(prompt_parts)

        # 8. 👇 核心修改：智能清理冲突词汇 👇
        # 仅在 standard 模式下清理，因为 mirror 模式需要手机倒影
        if selfie_style == "standard":
            phone_related_keywords = [
                r'\bholding\s+(a\s+)?(smart)?phone\b',  # 匹配 "holding a phone" 或 "holding smartphone"
                r'\bholding\s+(a\s+)?(smart)?phone\s+with\b',  # 匹配 "holding a phone with..."
                r'\bwith\s+(a\s+)?(smart)?phone\b',  # 匹配 "with a phone"
                r'\bphone\s+in\s+hand\b',  # 匹配 "phone in hand"
                r'\bphone\s+screen\b',  # 匹配 "phone screen"
                r'\bholding\s+(a\s+)?camera\b',  # 匹配 "holding a camera"
            ]
            
            # 执行清理
            for pattern in phone_related_keywords:
                final_prompt = re.sub(pattern, '', final_prompt, flags=re.IGNORECASE)
            
            # 清理多余的逗号和空格 (防止出现 "holding a, , phone" 这种残留)
            final_prompt = re.sub(r',\s*,+', ', ', final_prompt)
            final_prompt = re.sub(r'^,\s*', '', final_prompt)
            final_prompt = re.sub(r',\s*$', '', final_prompt)
            final_prompt = final_prompt.strip()

        # 9. 去重逻辑
        keywords = [kw.strip() for kw in final_prompt.split(',')]
        seen = set()
        unique_keywords = []
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower not in seen and kw:
                seen.add(kw_lower)
                unique_keywords.append(kw)

        final_prompt = ", ".join(unique_keywords)

        logger.info(f"{self.log_prefix} 自拍模式最终提示词: {final_prompt}") # 现在会显示所有提示词，方便找到问题
        return final_prompt

    def _get_selfie_reference_image(self) -> Optional[str]:
        """获取自拍参考图片的base64编码

        Returns:
            图片的base64编码，如果不存在则返回None
        """
        image_path = str(self.get_config("selfie.reference_image_path", "")).strip()
        if not image_path:
            return None

        try:
            # 处理相对路径（相对于插件目录）
            if not os.path.isabs(image_path):
                plugin_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                image_path = os.path.join(plugin_dir, image_path)

            if os.path.exists(image_path):
                with open(image_path, 'rb') as f:
                    image_data = f.read()
                image_base64 = base64.b64encode(image_data).decode('utf-8')
                logger.info(f"{self.log_prefix} 从文件加载自拍参考图片: {image_path}")
                return image_base64
            else:
                logger.warning(f"{self.log_prefix} 自拍参考图片文件不存在: {image_path}")
                return None
        except Exception as e:
            logger.error(f"{self.log_prefix} 加载自拍参考图片失败: {e}")
            return None

    async def _schedule_auto_recall_for_recent_message(self, model_config: Optional[Dict[str, Any]] = None):
        """安排最近发送消息的自动撤回

        通过查询数据库获取最近发送的消息ID，然后安排撤回任务

        Args:
            model_config: 当前使用的模型配置，用于检查撤回延时设置
        """
        # 检查全局开关
        global_enabled = bool(self.get_config("auto_recall.enabled", False))
        if not global_enabled:
            return

        # 检查模型的撤回延时，大于0才启用
        if not model_config:
            return

        delay_seconds = model_config.get("auto_recall_delay", 0)
        if delay_seconds <= 0:
            return

        # 获取模型ID用于检查运行时撤回状态
        model_id = None
        models_config = self.get_config("models", {})
        if isinstance(models_config, dict):
            for mid, config in models_config.items():
                # 通过模型名称匹配，避免字典比较问题
                if isinstance(config, dict) and config.get("model") == model_config.get("model"):
                    model_id = mid
                    break

        # 检查运行时撤回状态
        if model_id and not runtime_state.is_recall_enabled(self.chat_id, model_id, global_enabled):
            logger.info(f"{self.log_prefix} 模型 {model_id} 撤回已在当前聊天流禁用")
            return

        # 创建异步任务
        async def recall_task():
            try:
                # 等待足够时间让消息存储和 echo 回调完成（平台返回真实消息ID需要时间）
                await asyncio.sleep(4)

                # 查询最近发送的消息获取消息ID
                import time as time_module
                from src.plugin_system.apis import message_api
                from src.config.config import global_config

                current_time = time_module.time()
                # 查询最近10秒内本聊天中Bot发送的消息
                messages = message_api.get_messages_by_time_in_chat(
                    chat_id=self.chat_id,
                    start_time=current_time - 10,
                    end_time=current_time + 1,
                    limit=5,
                    limit_mode="latest"
                )

                # 找到Bot发送的图片消息
                bot_id = str(global_config.bot.qq_account)
                target_message_id = None

                for msg in messages:
                    if str(msg.user_info.user_id) == bot_id:
                        # 找到Bot发送的最新消息
                        mid = str(msg.message_id)
                        # 只使用纯数字的消息ID（QQ平台真实ID），跳过 send_api_xxx 格式的内部ID
                        if mid.isdigit():
                            target_message_id = mid
                            break
                        else:
                            logger.debug(f"{self.log_prefix} 跳过非平台消息ID: {mid}")

                if not target_message_id:
                    logger.warning(f"{self.log_prefix} 未找到有效的平台消息ID（需要纯数字格式）")
                    return

                logger.info(f"{self.log_prefix} 安排消息自动撤回，延时: {delay_seconds}秒，消息ID: {target_message_id}")

                # 等待指定时间后撤回
                await asyncio.sleep(delay_seconds)

                # 尝试多个撤回命令名（参考 recall_manager_plugin）
                DELETE_COMMAND_CANDIDATES = ["DELETE_MSG", "delete_msg", "RECALL_MSG", "recall_msg"]
                recall_success = False

                for cmd in DELETE_COMMAND_CANDIDATES:
                    try:
                        result = await self.send_command(
                            command_name=cmd,
                            args={"message_id": str(target_message_id)},
                            storage_message=False
                        )

                        # 检查返回结果
                        if isinstance(result, bool) and result:
                            recall_success = True
                            logger.info(f"{self.log_prefix} 消息自动撤回成功，命令: {cmd}，消息ID: {target_message_id}")
                            break
                        elif isinstance(result, dict):
                            status = str(result.get("status", "")).lower()
                            if status in ("ok", "success") or result.get("retcode") == 0 or result.get("code") == 0:
                                recall_success = True
                                logger.info(f"{self.log_prefix} 消息自动撤回成功，命令: {cmd}，消息ID: {target_message_id}")
                                break
                    except Exception as e:
                        logger.debug(f"{self.log_prefix} 撤回命令 {cmd} 失败: {e}")
                        continue

                if not recall_success:
                    logger.warning(f"{self.log_prefix} 消息自动撤回失败，消息ID: {target_message_id}，已尝试所有命令")

            except asyncio.CancelledError:
                logger.debug(f"{self.log_prefix} 自动撤回任务被取消")
            except Exception as e:
                logger.error(f"{self.log_prefix} 自动撤回失败: {e}")

        # 启动后台任务
        asyncio.create_task(recall_task())

    def _extract_description_from_message(self) -> str:
        """从用户消息中提取图片描述
        
        Returns:
            str: 提取的图片描述，如果无法提取则返回空字符串
        """
        if not self.action_message:
            return ""
            
        # 获取消息文本
        # 使用 getattr 安全获取 raw_message
        raw_msg = getattr(self.action_message, "raw_message", "")
        message_text = (self.action_message.processed_plain_text or
                       self.action_message.display_message or
                       raw_msg or "").strip()
        
        if not message_text:
            return ""
            
        import re
        
        # 移除常见的画图相关前缀
        patterns_to_remove = [
            r'^画',           # "画"
            r'^绘制',         # "绘制"
            r'^生成图片',     # "生成图片"
            r'^画图',         # "画图"
            r'^帮我画',       # "帮我画"
            r'^请画',         # "请画"
            r'^能不能画',     # "能不能画"
            r'^可以画',       # "可以画"
            r'^画一个',       # "画一个"
            r'^画一只',       # "画一只"
            r'^画张',         # "画张"
            r'^画幅',         # "画幅"
            r'^图[：:]',      # "图："或"图:"
            r'^生成图片[：:]', # "生成图片："或"生成图片:"
            r'^[：:]',        # 单独的冒号
        ]
        
        cleaned_text = message_text
        for pattern in patterns_to_remove:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
        
        # 移除常见的后缀
        suffix_patterns = [
            r'图片$',         # "图片"
            r'图$',           # "图"
            r'一下$',         # "一下"
            r'呗$',           # "呗"
            r'吧$',           # "吧"
        ]
        
        for pattern in suffix_patterns:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
        
        # 清理空白字符
        cleaned_text = cleaned_text.strip()
        
        # 如果清理后为空，返回原消息（可能是简单的描述）
        if not cleaned_text:
            cleaned_text = message_text
            
        # 限制长度，避免过长的描述
        if len(cleaned_text) > 100:
            cleaned_text = cleaned_text[:100]
            
        return cleaned_text
