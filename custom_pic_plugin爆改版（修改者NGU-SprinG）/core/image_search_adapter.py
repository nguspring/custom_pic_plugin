import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class ImageSearchAdapter:
    """图片搜索适配器：使用内置的搜索引擎进行图片搜索"""

    # 缓存导入的引擎类
    _engines_cache = {
        "bing": None,
        "sogou": None,
        "duckduckgo": None,
    }

    @classmethod
    def _import_engines(cls):
        """导入内置的搜索引擎类"""
        if all(cls._engines_cache.values()):
            return  # 已经导入过了

        try:
            # 导入内置的搜索引擎类
            from .search_engines.bing import BingEngine
            from .search_engines.sogou import SogouEngine
            from .search_engines.duckduckgo import DuckDuckGoEngine
            
            cls._engines_cache["bing"] = BingEngine
            cls._engines_cache["sogou"] = SogouEngine
            cls._engines_cache["duckduckgo"] = DuckDuckGoEngine
            
            logger.info("[ImageSearchAdapter] 成功导入内置搜索引擎")
        except ImportError as e:
            logger.warning(f"[ImageSearchAdapter] 导入内置搜索引擎失败: {e}")


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
                # 构建配置
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
