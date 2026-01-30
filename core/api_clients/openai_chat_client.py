"""OpenAI Chat 格式 API 客户端
专门用于处理通过 chat/completions 接口生成图片的供应商 (如 Nano Banana, OpenRouter, Claude 等)
支持从混合文本或 Markdown 中提取图片 URL 或 Base64 数据
"""

import json
import re
import urllib.request
from typing import Dict, Any, Tuple, Optional

from .base_client import BaseApiClient, logger
from ..size_utils import pixel_size_to_gemini_aspect


class OpenAIChatClient(BaseApiClient):
    """基于 Chat Completion 的图片生成客户端"""

    format_name = "openai-chat"

    def _make_request(
        self,
        prompt: str,
        model_config: Dict[str, Any],
        size: str,
        strength: Optional[float] = None,
        input_image_base64: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """发送 chat/completions 请求并解析图片"""
        base_url = model_config.get("base_url", "").rstrip("/")
        api_key = model_config.get("api_key", "")
        model = model_config.get("model", "")

        # 默认为 chat/completions，如果 base_url 已经包含路径则尝试适配
        if "/chat/completions" not in base_url and not base_url.endswith("/completions"):
            endpoint = f"{base_url}/chat/completions"
        else:
            endpoint = base_url

        # 组装 prompt
        custom_prompt_add = model_config.get("custom_prompt_add", "")
        full_prompt = prompt + custom_prompt_add

        # 构造消息内容
        contents = [{"type": "text", "text": full_prompt}]

        # 如果有输入图片，添加图生图支持 (兼容 Vision 格式)
        if input_image_base64:
            image_data_uri = self._prepare_image_data_uri(input_image_base64)
            contents.append({"type": "image_url", "image_url": {"url": image_data_uri}})

        messages = [{"role": "user", "content": contents if len(contents) > 1 else full_prompt}]

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
        }

        # 适配 Gemini/Nano Banana 风格的参数
        image_config = self._build_gemini_style_config(model_config, size)
        payload.update(image_config)

        # 基础参数
        seed = model_config.get("seed")
        if seed is not None and seed != -1:
            payload["seed"] = seed

        # 获取代理配置
        proxy_config = self._get_proxy_config()

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key.replace('Bearer ', '')}" if api_key else "",
        }

        logger.info(f"{self.log_prefix} (ChatImage) 发起请求: {model}, To: {endpoint}")

        req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")

        try:
            timeout = 600
            if proxy_config:
                proxy_handler = urllib.request.ProxyHandler(
                    {"http": proxy_config["http"], "https": proxy_config["https"]}
                )
                opener = urllib.request.build_opener(proxy_handler)
                urllib.request.install_opener(opener)
                timeout = proxy_config.get("timeout", 600)

            with urllib.request.urlopen(req, timeout=timeout) as response:
                response_status = response.status
                body_bytes = response.read()
                body_str = body_bytes.decode("utf-8")

                if 200 <= response_status < 300:
                    resp_json = json.loads(body_str)

                    # 1. 尝试从 choices 提取 (Chat 模式)
                    choices = resp_json.get("choices")
                    if isinstance(choices, list) and choices:
                        message = choices[0].get("message", {})
                        content = message.get("content")

                        # 执行深度提取
                        extracted = self._extract_image_from_content(content)
                        if extracted:
                            logger.info(f"{self.log_prefix} (ChatImage) 提取图片成功")
                            return True, extracted

                    # 2. 兜底尝试 OpenAI 标准图像格式 (以防万一)
                    if isinstance(resp_json.get("data"), list) and resp_json["data"]:
                        first = resp_json["data"][0]
                        if "b64_json" in first:
                            return True, first["b64_json"]
                        if "url" in first:
                            return True, first["url"]

                    logger.error(
                        f"{self.log_prefix} (ChatImage) 响应中未找到可识别的图片数据. Preview: {body_str[:200]}"
                    )
                    return False, "未能从回复中提取到图片信息"
                else:
                    logger.error(
                        f"{self.log_prefix} (ChatImage) API 请求失败 (HTTP {response_status}). Body: {body_str[:200]}"
                    )
                    return False, f"API 请求失败 (状态码 {response_status})"

        except Exception as e:
            logger.error(f"{self.log_prefix} (ChatImage) 请求异常: {e!r}", exc_info=True)
            return False, f"网络请求异常: {str(e)[:100]}"

    def _build_gemini_style_config(self, model_config: Dict[str, Any], size: Optional[str] = None) -> Dict[str, Any]:
        """适配 Nano Banana 等所需的 Gemini 风格参数"""
        fixed_size_enabled = model_config.get("fixed_size_enabled", False)
        default_size = model_config.get("default_size", "").strip()
        llm_original_size = size or model_config.get("_llm_original_size", "").strip() or None

        config = {}
        aspect_ratio = None
        resolution = None

        if not fixed_size_enabled:
            # 动态模式
            if llm_original_size:
                aspect_ratio = pixel_size_to_gemini_aspect(llm_original_size, self.log_prefix) or "1:1"
            else:
                aspect_ratio = "1:1"
        else:
            # 固定尺寸模式解析
            if default_size.startswith("-"):  # 仅分辨率
                resolution = default_size[1:].strip().upper()
                aspect_ratio = "1:1"
            elif "-" in default_size:  # 宽高比-分辨率
                parts = default_size.split("-", 1)
                aspect_ratio = parts[0].strip()
                resolution = parts[1].strip().upper()
            elif ":" in default_size:  # 仅宽高比
                aspect_ratio = default_size
            elif "x" in default_size.lower():  # 像素转宽高比
                aspect_ratio = pixel_size_to_gemini_aspect(default_size, self.log_prefix) or "1:1"

        if aspect_ratio:
            config["image_aspect_ratio"] = aspect_ratio
        if resolution:
            config["image_resolution"] = resolution

        return config

    def _extract_image_from_content(self, content: Any) -> Optional[str]:
        """从各种格式的 content 字段深度提取图片"""
        if not content:
            return None

        # 如果是列表格式 (OpenRouter/Vision 风格)
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                # 情况 A: 直接包含 image_url
                if item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url")
                    if url:
                        return url
                # 情况 B: 包含文本，从文本中提取
                if item.get("type") == "text":
                    res = self._extract_from_text(item.get("text", ""))
                    if res:
                        return res
            return None

        # 如果是字符串 (通用 Chat 模式)
        if isinstance(content, str):
            return self._extract_from_text(content)

        return None

    def _extract_from_text(self, text: str) -> Optional[str]:
        """多策略提取算法"""
        if not text:
            return None

        # 策略 1: Markdown 图片标签 ![alt](url_or_base64)
        md_match = re.search(r"!\[.*?\]\((.*?)\)", text, re.DOTALL)
        if md_match:
            candidate = md_match.group(1).strip()
            # 移除可能的引号
            candidate = candidate.strip("'\"")
            if self._is_valid_image_source(candidate):
                return candidate

        # 策略 2: 查找 Data URI (data:image/...)
        data_uri_match = re.search(r"data:image/[^;]+;base64,[\w+/=]+", text)
        if data_uri_match:
            return data_uri_match.group(0)

        # 策略 3: 基于特征检测纯 Base64 (通常占据大部分文本或在特定标记内)
        # 寻找常见的图片 Base64 起始特征
        prefixes = ("/9j/", "iVBORw", "UklGR", "R0lGOD")
        for pref in prefixes:
            start_idx = text.find(pref)
            if start_idx != -1:
                # 尝试提取后续合法的 base64 字符
                # Base64 包含 A-Z, a-z, 0-9, +, /, = 以及换行
                remainder = text[start_idx:]
                # 贪婪匹配直到非 base64 字符
                match = re.match(r"[A-Za-z0-9+/=\s\n\r]+", remainder)
                if match:
                    b64_candidate = match.group(0).replace("\n", "").replace("\r", "").replace(" ", "")
                    # 长度校验，图片通常比较长
                    if len(b64_candidate) > 100:
                        return b64_candidate

        # 策略 4: 查找普通 URL
        url_match = re.search(r"https?://\S+", text)
        if url_match:
            url = url_match.group(0).rstrip(").,> \"'")
            return url

        return None

    def _is_valid_image_source(self, s: str) -> bool:
        """检查提取到的字符串是否像是合法的图片源"""
        if not s:
            return False
        if s.startswith("http"):
            return True
        if s.startswith("data:image"):
            return True
        # Base64 特征检测
        return any(s.startswith(p) for p in ("/9j/", "iVBORw", "UklGR", "R0lGOD"))
