import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class ImageSearchAdapter:
    """图片搜索适配器：动态导入联网搜索插件的搜索引擎"""

    # 缓存导入的引擎类
    _engines_cache = {
        "bing": None,
        "sogou": None,
        "duckduckgo": None,
    }

    @classmethod
    def _import_engines(cls):
        """动态导入搜索引擎类"""
        if all(cls._engines_cache.values()):
            return  # 已经导入过了

        try:
            # 尝试导入联网搜索插件的引擎类
            from plugins.google_search_plugin.search_engines.bing import BingEngine
            from plugins.google_search_plugin.search_engines.sogou import SogouEngine
            from plugins.google_search_plugin.search_engines.duckduckgo import DuckDuckGoEngine
            
            cls._engines_cache["bing"] = BingEngine
            cls._engines_cache["sogou"] = SogouEngine
            cls._engines_cache["duckduckgo"] = DuckDuckGoEngine
            
            logger.info("[ImageSearchAdapter] 成功导入联网搜索插件的搜索引擎")
        except ImportError as e:
            logger.warning(f"[ImageSearchAdapter] 导入联网搜索插件引擎失败: {e}")
 

    @classmethod
    async def search(cls, keyword: str, max_results: int = 1) -> Optional[str]:
        """
        搜索关键词，返回第一张图片的URL
        
        Args:
            keyword: 搜索关键词
            max_results: 最多返回多少张
            
        Returns:
            图片URL，失败返回 None
        """
        cls._import_engines()
        
        # 构建查询词，加上 "official art" 提高找到高质量图片的概率
        query = f"{keyword} official art character design"
        logger.info(f"[ImageSearchAdapter] 正在搜索图片: {query}")
        
        # 按优先级尝试搜索引擎
        engines_order = ["bing", "sogou", "duckduckgo"]
        
        for engine_name in engines_order:
            engine_class = cls._engines_cache.get(engine_name)
            if not engine_class:
                continue
                
            try:
                # 处理不同的引擎实现
                if engine_class == "_ddgs":
                    # DuckDuckGo 直接实现
                    image_url = await cls._search_with_ddgs(query, max_results)
                    if image_url:
                        return image_url
                else:
                    # 联网搜索插件的引擎类
                    # 需要构建配置
                    config = {
                        "timeout": 20,
                        "proxy": "",  # 可以从配置读取
                        "max_results": max_results
                    }
                    
                    if engine_name == "bing":
                        config["region"] = "zh-CN"
                    elif engine_name == "sogou":
                        pass  # 无需额外配置
                    elif engine_name == "duckduckgo":
                        config["region"] = "wt-wt"
                        config["backend"] = "auto"
                        config["safesearch"] = "moderate"
                    
                    engine = engine_class(config)
                    results = await engine.search_images(query, max_results)
                    
                    if results and len(results) > 0:
                        # 提取第一张图片的URL
                        # 根据联网搜索插件的代码，返回的是 List[Dict] 或 List[SearchResult]
                        first_result = results[0]
                        image_url = first_result.get("image") if isinstance(first_result, dict) else getattr(first_result, "image", None)
                        
                        if image_url:
                            logger.info(f"[ImageSearchAdapter] 使用 {engine_name} 找到图片: {image_url}")
                            return image_url
            except Exception as e:
                logger.warning(f"[ImageSearchAdapter] {engine_name} 搜索失败: {e}")
                continue
        
        logger.warning(f"[ImageSearchAdapter] 所有搜索引擎均失败: {keyword}")
        return None

    @classmethod
    async def _search_with_ddgs(cls, query: str, max_results: int) -> Optional[str]:
        """使用 DuckDuckGo 直接搜索（备选方案）"""
        try:
            from duckduckgo_search import DDGS
            
            with DDGS() as ddgs:
                results = list(ddgs.images(query, max_results=max_results))
                
                if results and len(results) > 0:
                    image_url = results[0].get("image")
                    logger.info(f"[ImageSearchAdapter] DuckDuckGo 直接搜索成功: {image_url}")
                    return image_url
                
            return None
        except Exception as e:
            logger.error(f"[ImageSearchAdapter] DuckDuckGo 直接搜索失败: {e}")
            return None
