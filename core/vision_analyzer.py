import base64
import aiohttp
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class VisionAnalyzer:
    """视觉分析客户端，调用支持视觉的LLM提取特征"""

    def __init__(self, base_url: str, api_key: str, model_name: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name

    async def analyze_image(self, image_url: str) -> str:
        """
        下载图片并发送给视觉模型分析特征

        Args:
            image_url: 图片URL

        Returns:
            提取的英文提示词（如：red hair, white hat, ...）
        """
        try:
            # 1. 下载图片并转为Base64
            image_base64 = await self._download_and_encode(image_url)
            if not image_base64:
                return ""

            # 2. 构造请求体
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "请详细分析这张图片中的角色视觉特征。"
                                    "提取关键特征并转化为 Stable Diffusion 格式的英文提示词（Tag）。"
                                    "包括但不限于：发色、瞳色、发型、服装、配饰、姿势、背景风格等。"
                                    "只需返回提示词，不要包含任何解释性文字。"
                                ),
                            },
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                        ],
                    }
                ],
                "max_tokens": 300,
            }

            # 3. 发送请求
            api_endpoint = f"{self.base_url}/chat/completions"
            logger.info("[VisionAnalyzer] 正在请求视觉API分析图片...")

            async with aiohttp.ClientSession() as session:
                async with session.post(api_endpoint, headers=headers, json=payload) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(f"[VisionAnalyzer] API请求失败: {resp.status} - {error_text}")
                        return ""

                    data = await resp.json()
                    content = data["choices"][0]["message"]["content"].strip()
                    logger.info(f"[VisionAnalyzer] 分析成功，提取特征: {content[:100]}...")
                    return content

        except Exception as e:
            logger.error(f"[VisionAnalyzer] 分析过程出错: {e}", exc_info=True)
            return ""

    async def _download_and_encode(self, url: str) -> Optional[str]:
        """下载图片并转为base64"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status == 200:
                        image_bytes = await resp.read()
                        return base64.b64encode(image_bytes).decode("utf-8")
                    else:
                        logger.warning(f"[VisionAnalyzer] 下载图片失败: {resp.status}")
                        return None
        except Exception as e:
            logger.error(f"[VisionAnalyzer] 下载图片异常: {e}")
            return None
