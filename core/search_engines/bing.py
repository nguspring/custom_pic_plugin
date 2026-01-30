"""
Bing 图片搜索引擎实现
支持国内直接访问，无需代理
"""

import json
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import urlencode

# 延迟导入 BeautifulSoup，避免 IDE 类型检查警告
try:
    from bs4 import BeautifulSoup  # type: ignore[import-not-found]
except ImportError:
    BeautifulSoup = None  # type: ignore[assignment, misc]

from .base import BaseSearchEngine, SearchResult

logger = logging.getLogger(__name__)


class BingImageEngine(BaseSearchEngine):
    """Bing 图片搜索引擎实现"""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.base_urls: List[str] = ["https://cn.bing.com", "https://www.bing.com"]
        self.region: str = self.config.get("region", "zh-CN")

    async def search_images(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """执行Bing图片搜索（国内可直接访问，无需科学上网）

        Args:
            query: 搜索关键词
            num_results: 期望的图片数量

        Returns:
            图片搜索结果列表
        """
        if BeautifulSoup is None:
            logger.error("BeautifulSoup4 未安装，无法使用图片搜索功能")
            return []

        try:
            params: Dict[str, Any] = {
                "q": query,
                "first": 1,
                "count": min(num_results, 150),
                "cw": 1177,
                "ch": 826,
                "FORM": "HDRSC2",
            }

            # 尝试多个Bing图片搜索域名
            html: str = ""
            successful_base_url: str = ""
            for base_url in self.base_urls:
                try:
                    search_url = f"{base_url}/images/search?{urlencode(params)}"
                    logger.debug(f"请求Bing图片搜索URL: {search_url}")
                    html = await self._get_html(search_url)
                    if html and ("img_cont" in html or "iusc" in html):
                        successful_base_url = base_url
                        break
                except Exception as e:
                    logger.warning(f"Bing图片搜索域名 {base_url} 失败: {e}")
                    continue

            if not html:
                logger.warning(f"Bing图片搜索未获取到有效HTML: {query}")
                return []

            soup = BeautifulSoup(html, "html.parser")
            results: List[SearchResult] = []

            # 解析图片结果 - Bing图片搜索的HTML结构
            image_elements = soup.select("a.iusc")

            for elem in image_elements[:num_results]:
                try:
                    # 尝试从m属性获取JSON数据
                    m_attr = elem.get("m")
                    if m_attr:
                        try:
                            m_data: Dict[str, Any] = json.loads(m_attr)
                            image_url: str = m_data.get("murl", "")
                            thumbnail_url: str = m_data.get("turl", "")
                            title: str = m_data.get("t", "")

                            if image_url and image_url.startswith(("http://", "https://")):
                                results.append(
                                    SearchResult(
                                        title=title or query,
                                        url=image_url,
                                        image=image_url,
                                        thumbnail=thumbnail_url or image_url,
                                    )
                                )
                                continue
                        except json.JSONDecodeError:
                            # JSON解析失败，尝试备用方法
                            pass

                    # 备用解析：从img标签获取
                    img_elem = elem.find("img")
                    if img_elem:
                        image_url = img_elem.get("src") or img_elem.get("data-src") or ""
                        if image_url:
                            # 处理相对路径
                            if image_url.startswith("//"):
                                image_url = "https:" + image_url
                            elif image_url.startswith("/") and successful_base_url:
                                image_url = f"{successful_base_url}{image_url}"

                            if image_url.startswith(("http://", "https://")):
                                title = img_elem.get("alt") or query
                                results.append(
                                    SearchResult(title=title, url=image_url, image=image_url, thumbnail=image_url)
                                )
                except Exception as e:
                    logger.debug(f"解析Bing图片元素失败: {e}")
                    continue

            logger.info(f"Bing图片搜索找到 {len(results)} 张图片: {query}")
            return results[:num_results]

        except Exception as e:
            logger.error(f"Bing图片搜索错误: {query} - {e}", exc_info=True)
            return []
