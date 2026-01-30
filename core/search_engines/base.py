"""
图片搜索引擎基类
独立实现，不依赖外部插件
"""

import random
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

# 延迟导入 aiohttp，避免 IDE 类型检查警告
try:
    import aiohttp  # type: ignore[import-not-found]
except ImportError:
    aiohttp = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# HTTP 请求头
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36",
    "Accept": "*/*",
    "Connection": "keep-alive",
    "Accept-Language": "en-GB,en;q=0.5",
}

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36",
]


@dataclass
class SearchResult:
    """搜索结果数据类"""

    title: str
    url: str
    snippet: str = ""
    image: str = ""  # 图片URL
    thumbnail: str = ""  # 缩略图URL


class BaseSearchEngine:
    """搜索引擎基类"""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = config or {}
        self.timeout: int = self.config.get("timeout", 15)
        self.max_results: int = self.config.get("max_results", 5)
        self.headers: Dict[str, str] = HEADERS.copy()
        self.proxy: Optional[str] = self.config.get("proxy")

    async def _get_html(self, url: str, data: Optional[Dict[str, Any]] = None) -> str:
        """获取HTML内容

        Args:
            url: 目标URL
            data: POST数据（可选）

        Returns:
            HTML字符串
        """
        if aiohttp is None:
            logger.error("aiohttp 未安装，无法执行HTTP请求")
            return ""

        headers = self.headers.copy()
        headers["Referer"] = url
        headers["User-Agent"] = random.choice(USER_AGENTS)

        try:
            async with aiohttp.ClientSession() as session:
                if data:
                    async with session.post(
                        url,
                        headers=headers,
                        data=data,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                        proxy=self.proxy,
                    ) as resp:
                        resp.raise_for_status()
                        return await resp.text()
                else:
                    async with session.get(
                        url, headers=headers, timeout=aiohttp.ClientTimeout(total=self.timeout), proxy=self.proxy
                    ) as resp:
                        resp.raise_for_status()
                        return await resp.text()
        except Exception as e:
            logger.error(f"获取HTML失败: {url} - {e}")
            return ""

    def tidy_text(self, text: str) -> str:
        """清理文本

        Args:
            text: 原始文本

        Returns:
            清理后的文本
        """
        if not text:
            return ""

        # 常规文本清理
        text = text.strip().replace("\n", " ").replace("\r", " ")

        # 合并多个空格为单个空格
        while "  " in text:
            text = text.replace("  ", " ")

        return text

    async def search_images(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """搜索图片（子类需要实现）

        Args:
            query: 搜索关键词
            num_results: 期望的结果数量

        Returns:
            搜索结果列表
        """
        raise NotImplementedError("子类需要实现 search_images 方法")
