from typing import List, Tuple, Type, Dict, Any, Optional
import os

from src.plugin_system.base.base_plugin import BasePlugin
from src.plugin_system.base.component_types import ComponentInfo
from src.plugin_system import register_plugin
from src.plugin_system.base.config_types import (
    ConfigField,
    ConfigSection,
    ConfigLayout,
    ConfigTab,
)

from .core.pic_action import CustomPicAction
from .core.pic_command import PicGenerationCommand, PicConfigCommand, PicStyleCommand
from .core.config_manager import EnhancedConfigManager
from .core.auto_selfie_task import AutoSelfieTask


@register_plugin
class CustomPicPlugin(BasePlugin):
    """统一的多模型图片生成插件，支持文生图和图生图"""

    # 插件基本信息
    plugin_name: str = "custom_pic_plugin"  # type: ignore[assignment]
    plugin_version: str = "3.5.0-beta.14"
    plugin_author: str = "Ptrel，Rabbit，saberlights Kiuon，nguspring"
    enable_plugin: bool = True  # type: ignore[assignment]
    dependencies: List[str] = []  # type: ignore[assignment]
    python_dependencies: List[str] = ["aiohttp", "beautifulsoup4"]  # type: ignore[assignment]
    config_file_name: str = "config.toml"  # type: ignore[assignment]

    # 配置节元数据
    config_section_descriptions = {
        "plugin": ConfigSection(
            title="插件启用配置",
            icon="info",
            order=1
        ),
        "generation": ConfigSection(
            title="图片生成默认配置",
            icon="image",
            order=2
        ),
        "components": ConfigSection(
            title="组件启用配置",
            icon="puzzle-piece",
            order=3
        ),
        "proxy": ConfigSection(
            title="代理设置",
            icon="globe",
            order=4
        ),
        "cache": ConfigSection(
            title="结果缓存配置",
            icon="database",
            order=5
        ),
        "selfie": ConfigSection(
            title="自拍模式配置",
            icon="camera",
            order=6
        ),
        "auto_recall": ConfigSection(
            title="自动撤回配置",
            icon="trash",
            order=7
        ),
        "auto_selfie": ConfigSection(
            title="定时自拍配置",
            description="Bot会根据LLM智能日程动态发送自拍。v3.5.0起统一为smart模式，旧模式(interval/times/hybrid)自动升级",
            icon="clock",
            order=7
        ),
        "prompt_optimizer": ConfigSection(
            title="提示词优化器",
            description="使用 MaiBot 主 LLM 将用户描述优化为专业绘画提示词",
            icon="wand-2",
            order=8
        ),
        "search_reference": ConfigSection(
            title="智能参考搜索配置",
            description="自动搜索角色图片并提取特征，解决模型不认识角色的问题",
            icon="search",
            order=9
        ),
        "styles": ConfigSection(
            title="风格定义",
            description="预设风格的提示词。添加更多风格请直接编辑 config.toml，格式：风格英文名 = \"提示词\"",
            icon="palette",
            order=10
        ),
        "style_aliases": ConfigSection(
            title="风格别名",
            description="风格的中文别名映射。添加更多别名请直接编辑 config.toml",
            icon="tag",
            order=11
        ),
        "logging": ConfigSection(
            title="日志配置",
            icon="file-text",
            collapsed=True,
            order=12
        ),
        "models": ConfigSection(
            title="多模型配置",
            description="添加更多模型请直接编辑 config.toml，复制 [models.model1] 整节并改名为 model2、model3 等",
            icon="cpu",
            order=13
        ),
        "models.model1": ConfigSection(title="模型1配置", icon="box", order=14),
        "models.model2": ConfigSection(title="模型2配置", icon="box", order=15),
        "models.model3": ConfigSection(title="模型3配置", icon="box", order=16),
        "models.model4": ConfigSection(title="模型4配置", icon="box", order=17),
        "models.model5": ConfigSection(title="模型5配置", icon="box", order=18),
        "models.model6": ConfigSection(title="模型6配置", icon="box", order=19),
        "models.model7": ConfigSection(title="模型7配置", icon="box", order=20),
    }

    # 自定义布局：标签页
    config_layout = ConfigLayout(
        type="tabs",
        tabs=[
            ConfigTab(
                id="basic",
                title="基础设置",
                sections=["plugin", "generation", "components"],
                icon="settings"
            ),
            ConfigTab(
                id="network",
                title="网络配置",
                sections=["proxy", "cache"],
                icon="wifi"
            ),
            ConfigTab(
                id="features",
                title="功能配置",
                sections=["selfie", "auto_recall", "auto_selfie", "prompt_optimizer", "search_reference"],
                icon="zap"
            ),
            ConfigTab(
                id="styles",
                title="风格管理",
                sections=["styles", "style_aliases"],
                icon="palette"
            ),
            ConfigTab(
                id="models",
                title="模型管理",
                sections=["models", "models.model1", "models.model2", "models.model3", "models.model4", "models.model5", "models.model6", "models.model7"],
                icon="cpu"
            ),
            ConfigTab(
                id="advanced",
                title="高级",
                sections=["logging"],
                icon="terminal",
                badge="Dev"
            ),
        ]
    )

    # 配置Schema
    config_schema: Dict[str, Dict[str, ConfigField]] = {  # type: ignore[assignment]
        "plugin": {
            "name": ConfigField(
                type=str,
                default="custom_pic_plugin",
                description="智能多模型图片生成插件，支持文生图/图生图自动识别",
                required=True,
                disabled=True,
                order=1
            ),
            "config_version": ConfigField(
                type=str,
                default="3.5.0-beta.14",
                description="插件配置版本号",
                disabled=True,
                order=2
            ),
            "enabled": ConfigField(
                type=bool,
                default=False,
                description="是否启用插件，开启后可使用画图和风格转换功能",
                order=3
            )
        },
        "generation": {
            "default_model": ConfigField(
                type=str,
                default="model1",
                description="默认使用的模型ID，用于智能图片生成。支持文生图和图生图自动识别",
                placeholder="model1",
                order=1
            ),
        },
        "cache": {
            "enabled": ConfigField(
                type=bool,
                default=False,
                description="是否启用结果缓存，相同参数的请求会复用之前的结果",
                order=1
            ),
            "max_size": ConfigField(
                type=int,
                default=10,
                description="最大缓存数量，超出后删除最旧的缓存",
                min=1,
                max=100,
                depends_on="cache.enabled",
                depends_value=True,
                order=2
            ),
        },
        "components": {
            "enable_unified_generation": ConfigField(
                type=bool,
                default=True,
                description="是否启用智能图片生成Action，支持文生图和图生图自动识别",
                order=1
            ),
            "enable_pic_command": ConfigField(
                type=bool,
                default=True,
                description="是否启用风格化图生图Command功能，支持/dr <风格>命令",
                order=2
            ),
            "enable_pic_config": ConfigField(
                type=bool,
                default=True,
                description="是否启用模型配置管理命令，支持/dr list、/dr set等",
                order=3
            ),
            "enable_pic_style": ConfigField(
                type=bool,
                default=True,
                description="是否启用风格管理命令，支持/dr styles、/dr style等",
                order=4
            ),
            "pic_command_model": ConfigField(
                type=str,
                default="model1",
                description="Command组件使用的模型ID，可通过/dr set命令动态切换",
                placeholder="model1",
                order=5
            ),
            "enable_debug_info": ConfigField(
                type=bool,
                default=False,
                description="是否启用调试信息显示，关闭后仅显示图片结果和错误信息",
                order=6
            ),
            "enable_verbose_debug": ConfigField(
                type=bool,
                default=False,
                description="是否启用详细调试信息，启用后会发送完整的调试信息以及打印完整的 POST 报文",
                order=7
            ),
            "admin_users": ConfigField(
                type=list,
                default=[],
                description="有权限使用配置管理命令的管理员用户列表，请填写字符串形式的用户ID",
                placeholder="[\"用户ID1\", \"用户ID2\"]",
                order=8
            ),
            "max_retries": ConfigField(
                type=int,
                default=2,
                description="API调用失败时的重试次数，建议2-5次。设置为0表示不重试",
                min=0,
                max=10,
                order=9
            )
        },
        "logging": {
            "level": ConfigField(
                type=str,
                default="INFO",
                description="日志记录级别，DEBUG显示详细信息",
                choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                order=1
            ),
            "prefix": ConfigField(
                type=str,
                default="[unified_pic_Plugin]",
                description="日志前缀标识",
                placeholder="[插件名]",
                order=2
            )
        },
        "proxy": {
            "enabled": ConfigField(
                type=bool,
                default=False,
                description="是否启用代理。开启后所有API请求将通过代理服务器",
                order=1
            ),
            "url": ConfigField(
                type=str,
                default="http://127.0.0.1:7890",
                description="代理服务器地址，格式：http://host:port。支持HTTP/HTTPS/SOCKS5代理",
                placeholder="http://127.0.0.1:7890",
                depends_on="proxy.enabled",
                depends_value=True,
                order=2
            ),
            "timeout": ConfigField(
                type=int,
                default=60,
                description="代理连接超时时间（秒），建议30-120秒",
                min=10,
                max=300,
                depends_on="proxy.enabled",
                depends_value=True,
                order=3
            )
        },
        "styles": {
            "hint": ConfigField(
                type=str,
                default="添加更多风格：编辑 config.toml，在 [styles] 节下添加 风格英文名 = \"提示词\"",
                description="配置说明",
                disabled=True,
                order=0
            ),
            "cartoon": ConfigField(
                type=str,
                default="cartoon style, anime style, colorful, vibrant colors, clean lines",
                description="卡通风格提示词",
                input_type="textarea",
                rows=3,
                order=1
            ),
            "reality": ConfigField(
                type=str,
                default="photorealistic, professional photography, realistic details, soft natural light, cinematic lighting, proper leg anatomy, realistic proportions, intricate fabric textures",
                description="写实风格提示词",
                input_type="textarea",
                rows=3,
                order=2
            )
        },
        "style_aliases": {
            "hint": ConfigField(
                type=str,
                default="添加更多别名：编辑 config.toml，在 [style_aliases] 节下添加 风格英文名 = \"中文别名\"",
                description="配置说明",
                disabled=True,
                order=0
            ),
            "cartoon": ConfigField(
                type=str,
                default="卡通",
                description="cartoon 风格的中文别名，支持多别名用逗号分隔",
                placeholder="卡通,动漫",
                order=1
            ),
            "reality": ConfigField(
                type=str,
                default="现实, 真实",
                description="reality 风格的中文别名，支持多别名用逗号分隔",
                placeholder="现实, 真实",
                order=2
            )
        },
        "selfie": {
            "enabled": ConfigField(
                type=bool,
                default=True,
                description="是否启用自拍模式功能",
                order=1
            ),
            "reference_image_path": ConfigField(
                type=str,
                default="",
                description="自拍参考图片路径（相对于插件目录或绝对路径）。配置后自动使用图生图模式，留空则使用纯文生图。若模型不支持图生图会自动回退",
                placeholder="images/reference.png",
                depends_on="selfie.enabled",
                depends_value=True,
                order=2
            ),
            "prompt_prefix": ConfigField(
                type=str,
                default="",
                description="自拍模式专用提示词前缀。用于添加Bot的默认形象特征（发色、瞳色、服装风格等）。例如：'blue hair, red eyes, school uniform, 1girl'",
                input_type="textarea",
                rows=2,
                placeholder="blue hair, red eyes, school uniform, 1girl",
                depends_on="selfie.enabled",
                depends_value=True,
                order=3
            ),
            "negative_prompt_standard": ConfigField(
                type=str,
                default="phone, smartphone, mobile device, camera, selfie stick, visible electronic device, phone in hand, hands holding device, device screen, fingers on phone",
                description="标准自拍模式（standard）专用的负面提示词，会叠加在模型默认负面提示词上。标准自拍：手机在画框外，禁止手机出现",
                input_type="textarea",
                rows=2,
                placeholder="phone, smartphone, camera...",
                depends_on="selfie.enabled",
                depends_value=True,
                order=4
            ),
            "negative_prompt_mirror": ConfigField(
                type=str,
                default="selfie stick",
                description="对镜自拍模式（mirror）专用的负面提示词，会叠加在模型默认负面提示词上。对镜自拍：需要手持设备拍照，允许拍照设备出现",
                input_type="textarea",
                rows=2,
                placeholder="selfie stick",
                depends_on="selfie.enabled",
                depends_value=True,
                order=5
            ),
            "negative_prompt": ConfigField(
                type=str,
                default="",
                description="自拍模式通用的负面提示词（可选），会叠加在模型默认负面提示词上",
                input_type="textarea",
                rows=2,
                placeholder="",
                depends_on="selfie.enabled",
                depends_value=True,
                order=6
            ),
            "scene_mirror": ConfigField(
                type=str,
                default="mirror selfie, reflection in mirror, holding phone in hand, phone visible, arm slightly bent, looking at mirror, indoor scene, soft lighting, high quality",
                description="对镜自拍模式（mirror）的场景描述",
                input_type="textarea",
                rows=2,
                depends_on="selfie.enabled",
                depends_value=True,
                order=7
            ),
            "scene_standard": ConfigField(
                type=str,
                default="selfie, front camera view, (cowboy shot or full body shot or upper body), looking at camera, slight high angle selfie",
                description="标准自拍模式（standard）的场景描述",
                input_type="textarea",
                rows=2,
                depends_on="selfie.enabled",
                depends_value=True,
                order=8
            )
        },
        "auto_recall": {
            "enabled": ConfigField(
                type=bool,
                default=False,
                description="是否启用自动撤回功能（总开关）。关闭后所有模型的撤回都不生效",
                order=1
            )
        },
        # ================================================================================
        # 定时自拍配置 - 配置项按逻辑分组，清晰易懂
        # ================================================================================
        "auto_selfie": {
            # ==================== 1. 基础开关 ====================
            "enabled": ConfigField(
                type=bool,
                default=False,
                description="【总开关】是否启用定时自拍功能。开启后麦麦会按照schedule_times设定的时间点自动发送自拍。需要先配置好chat_id_list指定发送目标",
                order=1
            ),
            
            # ==================== 2. 发送时间设置 ====================
            "schedule_times": ConfigField(
                type=list,
                default=["07:30", "09:00", "10:30", "12:00", "14:00", "16:00", "18:00", "20:00", "22:00"],
                description="【核心配置】每天发送自拍的时间点列表。格式：24小时制HH:MM。LLM会根据每个时间点自动生成对应的场景（如08:00生成起床场景）。建议配置5-9个时间点覆盖一天的活动。示例：[\"08:00\", \"12:00\", \"18:00\", \"21:00\"]",
                placeholder='["08:00", "12:00", "18:00", "21:00"]',
                depends_on="auto_selfie.enabled",
                depends_value=True,
                order=2
            ),
            "sleep_mode_enabled": ConfigField(
                type=bool,
                default=True,
                description="是否启用睡眠模式。开启后在sleep_start_time到sleep_end_time期间不会发送自拍，让麦麦也有休息时间",
                depends_on="auto_selfie.enabled",
                depends_value=True,
                order=3
            ),
            "sleep_start_time": ConfigField(
                type=str,
                default="23:00",
                description="睡眠开始时间，格式：24小时制HH:MM。从此时间开始进入睡眠模式。示例：23:00",
                placeholder="23:00",
                depends_on="auto_selfie.sleep_mode_enabled",
                depends_value=True,
                order=4
            ),
            "sleep_end_time": ConfigField(
                type=str,
                default="07:00",
                description="睡眠结束时间，格式：24小时制HH:MM。到此时间结束睡眠模式。示例：07:00",
                placeholder="07:00",
                depends_on="auto_selfie.sleep_mode_enabled",
                depends_value=True,
                order=5
            ),
            
            # ==================== 3. 发送目标设置 ====================
            "list_mode": ConfigField(
                type=str,
                default="whitelist",
                description="名单模式选择。whitelist（白名单）=只向chat_id_list中的聊天发送自拍；blacklist（黑名单）=向所有聊天发送但排除chat_id_list中的",
                choices=["whitelist", "blacklist"],
                depends_on="auto_selfie.enabled",
                depends_value=True,
                order=6
            ),
            "chat_id_list": ConfigField(
                type=list,
                default=[],
                description="【核心配置】自拍发送目标列表。⚠️格式说明：在config.toml文件中填写时，每个ID必须加双引号！多个ID用逗号隔开。群聊格式：\"qq:群号:group\"，私聊格式：\"qq:用户QQ号:private\"。示例：[\"qq:123456789:group\", \"qq:987654321:private\"]。通过WebUI修改时无需手动加引号",
                placeholder='["qq:123456789:group", "qq:987654321:private"]',
                depends_on="auto_selfie.enabled",
                depends_value=True,
                order=7
            ),
            
            # ==================== 4. 日程生成设置 ====================
            "schedule_min_entries": ConfigField(
                type=int,
                default=4,
                description="每天日程的最少条目数。LLM生成日程时会确保至少有这么多条活动安排。建议4-6条",
                min=1,
                max=20,
                depends_on="auto_selfie.enabled",
                depends_value=True,
                order=8
            ),
            "schedule_max_entries": ConfigField(
                type=int,
                default=8,
                description="每天日程的最多条目数。限制LLM生成日程时的最大条目，避免日程过于繁忙。建议6-10条",
                min=1,
                max=20,
                depends_on="auto_selfie.enabled",
                depends_value=True,
                order=9
            ),
            "schedule_generator_model": ConfigField(
                type=str,
                default="",
                description="【进阶配置】日程生成使用的LLM模型ID。留空则自动使用MaiBot的replyer模型。注意：这是MaiBot主配置中的文本模型ID（如model_utils_1），不是本插件的绘图模型ID",
                placeholder="留空使用默认",
                depends_on="auto_selfie.enabled",
                depends_value=True,
                order=10
            ),
            "schedule_persona_enabled": ConfigField(
                type=bool,
                default=True,
                description="是否启用日程人设注入。开启后LLM会根据schedule_persona_text中描述的角色身份来生成日程，让活动安排更符合人设",
                depends_on="auto_selfie.enabled",
                depends_value=True,
                order=11
            ),
            "schedule_persona_text": ConfigField(
                type=str,
                default="是一个大二女大学生",
                description="日程人设描述。描述角色的身份背景，LLM会据此生成符合人设的日程。示例：是一个大学生（会生成上课、图书馆等场景）；是一个上班族（会生成通勤、办公室等场景）；是一个自由职业者（时间更自由灵活）",
                input_type="textarea",
                rows=2,
                placeholder="是一个大二女大学生，在市区租房住",
                hint="描述角色的身份背景，影响日程中的活动场景",
                depends_on="auto_selfie.schedule_persona_enabled",
                depends_value=True,
                order=12
            ),
            "schedule_lifestyle": ConfigField(
                type=str,
                default="作息规律，喜欢宅家但偶尔也会出门",
                description="生活习惯描述。控制日程的整体风格，如活动频率、外出习惯等。示例：早睡早起型/夜猫子型；宅家党/户外爱好者；社恐/社牛",
                input_type="textarea",
                rows=2,
                placeholder="作息规律，喜欢宅家追剧，偶尔和朋友出去逛街",
                hint="描述角色的生活习惯，让日程更符合人设",
                depends_on="auto_selfie.schedule_persona_enabled",
                depends_value=True,
                order=13
            ),
            
            # ==================== 5. 间隔补充发送（v3.5.0-beta.14：暂时禁用） ====================
            "enable_interval_supplement": ConfigField(
                type=bool,
                default=False,
                description="【暂时禁用】是否启用间隔补充发送。开启后会在schedule_times时间点之外随机触发自拍，让发送时间更自然。⚠️v3.5.0-beta.14临时禁用此功能以修复\"就近条目\"策略导致的重复场景问题（如多次显示\"起床\"场景）。建议保持关闭",
                depends_on="auto_selfie.enabled",
                depends_value=True,
                order=14
            ),
            "interval_minutes": ConfigField(
                type=int,
                default=120,
                description="间隔补充的最小时间间隔（分钟）。两次补充发送之间至少间隔这么久。默认120分钟=2小时",
                min=30,
                max=480,
                depends_on="auto_selfie.enable_interval_supplement",
                depends_value=True,
                order=15
            ),
            "interval_probability": ConfigField(
                type=float,
                default=0.3,
                description="间隔补充的触发概率。每次达到间隔时间时，有此概率实际触发发送。0.3=30%概率。范围0.0-1.0",
                min=0.0,
                max=1.0,
                step=0.1,
                depends_on="auto_selfie.enable_interval_supplement",
                depends_value=True,
                order=16
            ),
            
            # ==================== 6. 图片生成设置 ====================
            "model_id": ConfigField(
                type=str,
                default="model1",
                description="定时自拍使用的绘图模型ID。对应本插件[models.xxx]配置节的模型。示例：model1、model2等",
                placeholder="model1",
                depends_on="auto_selfie.enabled",
                depends_value=True,
                order=17
            ),
            "selfie_style": ConfigField(
                type=str,
                default="standard",
                description="自拍风格选择。standard=标准自拍（模拟前置摄像头，手机不出现在画面中）；mirror=对镜自拍（模拟对着镜子拍照，可能看到手机）",
                choices=["standard", "mirror"],
                depends_on="auto_selfie.enabled",
                depends_value=True,
                order=18
            ),
            
            # ==================== 7. 配文设置 ====================
            "enable_narrative": ConfigField(
                type=bool,
                default=True,
                description="是否启用叙事系统。开启后配文会根据当天日程生成连贯的\"故事线\"，让一天的自拍像是在记录生活",
                depends_on="auto_selfie.enabled",
                depends_value=True,
                order=19
            ),
            "use_replyer_for_ask": ConfigField(
                type=bool,
                default=True,
                description="是否使用LLM动态生成配文。开启后会调用LLM根据场景生成自然的配文；关闭则使用ask_message中的固定文案或随机模板",
                depends_on="auto_selfie.enabled",
                depends_value=True,
                order=20
            ),
            "ask_message": ConfigField(
                type=str,
                default="",
                description="固定配文内容。仅当use_replyer_for_ask=false时生效。留空则从预设模板中随机选择",
                placeholder="你看这张照片怎么样？",
                depends_on="auto_selfie.use_replyer_for_ask",
                depends_value=False,
                order=21
            ),
            "caption_model_id": ConfigField(
                type=str,
                default="",
                description="【进阶配置】配文生成使用的LLM模型ID。留空则自动使用MaiBot的replyer模型。与schedule_generator_model相同，这是MaiBot主配置中的文本模型ID",
                placeholder="留空使用默认",
                depends_on="auto_selfie.enable_narrative",
                depends_value=True,
                order=22
            ),
            "caption_types": ConfigField(
                type=list,
                default=["narrative", "ask", "share", "monologue", "none"],
                description="【高级设置】启用的配文类型列表。narrative=叙事式（今天去了...）；ask=询问式（你看这个...怎么样？）；share=分享式（推荐这个...）；monologue=独白式（内心OS）；none=不加配文",
                placeholder='["narrative", "ask", "share", "monologue", "none"]',
                depends_on="auto_selfie.enable_narrative",
                depends_value=True,
                order=23
            ),
            "caption_weights": ConfigField(
                type=list,
                default=[0.35, 0.25, 0.25, 0.10, 0.05],
                description="【高级设置】各配文类型的权重。与caption_types一一对应。权重越高该类型配文出现概率越大。所有权重之和应为1.0",
                placeholder="[0.35, 0.25, 0.25, 0.10, 0.05]",
                depends_on="auto_selfie.enable_narrative",
                depends_value=True,
                order=24
            ),
            
            # ==================== 8. 配文人设注入 ====================
            "caption_persona_enabled": ConfigField(
                type=bool,
                default=True,
                description="是否启用配文人设注入。开启后LLM会根据caption_persona_text中的人设生成配文，让配文风格符合角色个性",
                depends_on="auto_selfie.enable_narrative",
                depends_value=True,
                order=25
            ),
            "caption_persona_text": ConfigField(
                type=str,
                default="是一个喜欢分享日常的女生",
                description="配文人设描述。描述角色的身份和性格特点，LLM会据此调整配文的语气和内容。示例：是一个大二女大学生，有点小傲娇但其实很热心；是一个宅家程序员，说话直接但善良",
                input_type="textarea",
                rows=3,
                placeholder="是一个大二女大学生，喜欢分享日常，有点小傲娇但其实很热心",
                hint="描述角色的身份、性格特点，让配文更符合人设",
                depends_on="auto_selfie.caption_persona_enabled",
                depends_value=True,
                order=26
            ),
            "caption_reply_style": ConfigField(
                type=str,
                default="语气自然，符合年轻人社交风格",
                description="配文表达风格指导。控制LLM生成配文时的语言风格。示例：俏皮可爱，喜欢用颜文字；正经认真，像在写日记；吐槽风格，经常自嘲",
                input_type="textarea",
                rows=2,
                placeholder="俏皮可爱，偶尔吐槽，用语轻松自然",
                hint="描述说话的语气、习惯，让配文风格更一致",
                depends_on="auto_selfie.caption_persona_enabled",
                depends_value=True,
                order=27
            )
        },
        "prompt_optimizer": {
            "enabled": ConfigField(
                type=bool,
                default=True,
                description="是否启用提示词优化器。开启后会使用 MaiBot 主 LLM 将用户描述优化为专业英文提示词",
                order=1
            ),
            "hint": ConfigField(
                type=str,
                default="优化器会自动将中文描述翻译并优化为专业的英文绘画提示词，提升生成效果。关闭后将直接使用用户原始描述。",
                description="功能说明",
                disabled=True,
                order=2
            )
        },
        "search_reference": {
            "hint": ConfigField(
                type=str,
                default="开启后，当用户提到冷门角色名时，插件会自动联网搜索该角色的参考图，并使用视觉AI提取特征（发色、服装等）合并到提示词中，尽量缓解绘图模型不认识角色的问题。注意：如果使用的绘图模型本身就能够联网或认识角色（如Gemini），则不必开启本功能。",
                description="功能说明",
                disabled=True,
                order=0
            ),
            "enabled": ConfigField(
                type=bool,
                default=False,
                description="是否启用智能参考搜索功能",
                order=1
            ),
            "vision_api_key": ConfigField(
                type=str,
                default="",
                description="用于识图的API Key。留空则使用MaiBot的vlm模型（视觉语言模型），填写后则使用下方自定义的vision_model配置",
                input_type="password",
                depends_on="search_reference.enabled",
                depends_value=True,
                order=2
            ),
            "vision_base_url": ConfigField(
                type=str,
                default="https://api.openai.com/v1",
                description="识图API地址（仅在vision_api_key不为空时生效）",
                depends_on="search_reference.enabled",
                depends_value=True,
                order=3
            ),
            "vision_model": ConfigField(
                type=str,
                default="gpt-4o",
                description="视觉模型名称（仅在vision_api_key不为空时生效）",
                depends_on="search_reference.enabled",
                depends_value=True,
                order=4
            )
        },
        "models": {
            "hint": ConfigField(
                type=str,
                default="添加更多模型：编辑 config.toml，复制 [models.model1] 整节并改名为 model2、model3 等",
                description="配置说明",
                disabled=True,
                order=1
            )
        },
        # 基础模型配置模板 - model1: Tongyi-MAI/Z-Image-Turbo (推荐，速度快质量好)
        "models.model1": {
            "name": ConfigField(type=str, default="Tongyi-MAI/Z-Image-Turbo", description="模型显示名称，在模型列表中展示，版本更新后请手动从 old 目录恢复配置", order=1),
            "base_url": ConfigField(type=str, default="https://api-inference.modelscope.cn/v1", description="API服务地址。示例: OpenAI=https://api.openai.com/v1, 硅基流动=https://api.siliconflow.cn/v1, 豆包=https://ark.cn-beijing.volces.com/api/v3, 魔搭=https://api-inference.modelscope.cn/v1, Gemini=https://generativelanguage.googleapis.com", required=True, placeholder="https://api.example.com/v1", order=2),
            "api_key": ConfigField(type=str, default="Bearer xxxxxxxxxxxxxxxxxxxxxx", description="API密钥。OpenAI/modelscope格式需'Bearer '前缀，豆包/Gemini格式无需前缀", input_type="password", required=True, placeholder="Bearer sk-xxx 或 sk-xxx", order=3),
            "format": ConfigField(type=str, default="modelscope", description="API格式。openai=通用格式，openai-chat=Chat接口生图(反重力反代Gemini等)，doubao=豆包，gemini=Gemini，modelscope=魔搭，shatangyun=砂糖云(NovelAI)，mengyuai=梦羽AI，zai=Zai(Gemini转发)", choices=["openai", "openai-chat", "gemini", "doubao", "modelscope", "shatangyun", "mengyuai", "zai"], order=4),
            "model": ConfigField(type=str, default="Tongyi-MAI/Z-Image-Turbo", description="模型名称。梦羽AI格式填写模型索引数字（如0、1、2）", placeholder="model-name 或 0", order=5),
            "fixed_size_enabled": ConfigField(type=bool, default=False, description="是否固定图片尺寸。开启后强制使用default_size，关闭则麦麦选择", order=6),
            "default_size": ConfigField(type=str, default="1024x1024", description="默认图片尺寸。OpenAI/豆包/魔搭格式填写如 1024x1024。Gemini格式填写宽高比如 16:9 或 16:9-2K，具体参考官方文档", placeholder="1024x1024 或 16:9-2K", order=7),
            "seed": ConfigField(type=int, default=-1, description="随机种子，固定值可确保结果可复现", min=-1, max=2147483647, order=8),
            "guidance_scale": ConfigField(type=float, default=2.5, description="指导强度。豆包推荐5.5，其他推荐2.5。越高越严格遵循提示词", min=0.0, max=20.0, step=0.5, order=9),
            "num_inference_steps": ConfigField(type=int, default=30, description="推理步数，影响质量和速度。推荐20-50", min=1, max=150, order=10),
            "watermark": ConfigField(type=bool, default=False, description="是否添加水印，豆包默认支持", order=11),
            "custom_prompt_add": ConfigField(type=str, default="", description="正面提示词增强，自动添加到用户描述后", input_type="textarea", rows=2, order=12),
            "negative_prompt_add": ConfigField(type=str, default="low quality, worst quality, bad quality, lowres, blurry, text, watermark, signature, extra arms, extra legs, extra hands, extra fingers, extra toes, missing fingers, bad anatomy, bad hands, bad proportions, extra thighs, extra calves, leg duplication, leg artifacts, logo, bubble, extra limbs", description="负面提示词，避免不良内容。豆包可留空但需保留引号", input_type="textarea", rows=2, order=13),
            "artist": ConfigField(type=str, default="", description="艺术家风格标签（砂糖云专用）。留空则不添加", order=14),
            "support_img2img": ConfigField(type=bool, default=True, description="该模型是否支持图生图功能，请根据API文档自行判断。设为false时会自动降级为文生图", order=15),
            "auto_recall_delay": ConfigField(type=int, default=0, description="自动撤回延时（秒）。大于0时启用撤回，0表示不撤回。需先在「自动撤回配置」中开启总开关", min=0, max=120, order=16),
        },
        "models.model2": {
            "name": ConfigField(type=str, default="QWQ114514123/WAI-illustrious-SDXL-v16", description="模型显示名称，在模型列表中展示，版本更新后请手动从 old 目录恢复配置"),
            "base_url": ConfigField(type=str, default="https://api-inference.modelscope.cn/v1", description="API服务地址。示例: OpenAI=https://api.openai.com/v1, 硅基流动=https://api.siliconflow.cn/v1, 豆包=https://ark.cn-beijing.volces.com/api/v3, 魔搭=https://api-inference.modelscope.cn/v1, Gemini=https://generativelanguage.googleapis.com"),
            "api_key": ConfigField(type=str, default="Bearer xxxxxxxxxxxxxxxxxxxxxx", description="API密钥。OpenAI/modelscope格式需'Bearer '前缀，豆包/Gemini格式无需前缀", input_type="password", placeholder="Bearer sk-xxx 或 sk-xxx"),
            "format": ConfigField(type=str, default="modelscope", description="API格式。openai=通用格式，openai-chat=Chat接口生图(反重力反代Gemini等)，doubao=豆包，gemini=Gemini，modelscope=魔搭，shatangyun=砂糖云(NovelAI)，mengyuai=梦羽AI，zai=Zai(Gemini转发)", choices=["openai", "openai-chat", "gemini", "doubao", "modelscope", "shatangyun", "mengyuai", "zai"]),
            "model": ConfigField(type=str, default="QWQ114514123/WAI-illustrious-SDXL-v16", description="模型名称。梦羽AI格式填写模型索引数字（如0、1、2）"),
            "fixed_size_enabled": ConfigField(type=bool, default=False, description="是否固定图片尺寸。开启后强制使用default_size，关闭则麦麦选择"),
            "default_size": ConfigField(type=str, default="1024x1024", description="默认图片尺寸。OpenAI/豆包/魔搭格式填写如 1024x1024。Gemini格式填写宽高比如 16:9 或 16:9-2K"),
            "seed": ConfigField(type=int, default=-1, description="随机种子，固定值可确保结果可复现"),
            "guidance_scale": ConfigField(type=float, default=2.5, description="指导强度。豆包推荐5.5，其他推荐2.5"),
            "num_inference_steps": ConfigField(type=int, default=30, description="推理步数，影响质量和速度。推荐20-50"),
            "watermark": ConfigField(type=bool, default=False, description="是否添加水印"),
            "custom_prompt_add": ConfigField(type=str, default="", description="正面提示词增强，自动添加到用户描述后", input_type="textarea"),
            "negative_prompt_add": ConfigField(type=str, default="low quality, worst quality, bad quality, lowres, blurry, text, watermark, signature, extra arms, extra legs, extra hands, extra fingers, extra toes, missing fingers, bad anatomy, bad hands, bad proportions, extra thighs, extra calves, leg duplication, leg artifacts, logo, bubble, extra limbs", description="负面提示词，避免不良内容", input_type="textarea"),
            "artist": ConfigField(type=str, default="", description="艺术家风格标签（砂糖云专用）"),
            "support_img2img": ConfigField(type=bool, default=True, description="该模型是否支持图生图功能"),
            "auto_recall_delay": ConfigField(type=int, default=0, description="自动撤回延时（秒）"),
        },
        "models.model3": {
            "name": ConfigField(type=str, default="ChenkinNoob/ChenkinNoob-XL-V0.2", description="模型显示名称，在模型列表中展示，版本更新后请手动从 old 目录恢复配置"),
            "base_url": ConfigField(type=str, default="https://api-inference.modelscope.cn/v1", description="API服务地址。示例: OpenAI=https://api.openai.com/v1, 硅基流动=https://api.siliconflow.cn/v1, 豆包=https://ark.cn-beijing.volces.com/api/v3, 魔搭=https://api-inference.modelscope.cn/v1, Gemini=https://generativelanguage.googleapis.com"),
            "api_key": ConfigField(type=str, default="Bearer xxxxxxxxxxxxxxxxxxxxxx", description="API密钥。OpenAI/modelscope格式需'Bearer '前缀，豆包/Gemini格式无需前缀", input_type="password", placeholder="Bearer sk-xxx 或 sk-xxx"),
            "format": ConfigField(type=str, default="modelscope", description="API格式。openai=通用格式，openai-chat=Chat接口生图(反重力反代Gemini等)，doubao=豆包，gemini=Gemini，modelscope=魔搭，shatangyun=砂糖云(NovelAI)，mengyuai=梦羽AI，zai=Zai(Gemini转发)", choices=["openai", "openai-chat", "gemini", "doubao", "modelscope", "shatangyun", "mengyuai", "zai"]),
            "model": ConfigField(type=str, default="ChenkinNoob/ChenkinNoob-XL-V0.2", description="模型名称。梦羽AI格式填写模型索引数字（如0、1、2）"),
            "fixed_size_enabled": ConfigField(type=bool, default=False, description="是否固定图片尺寸。开启后强制使用default_size，关闭则麦麦选择"),
            "default_size": ConfigField(type=str, default="832x1216", description="默认图片尺寸。OpenAI/豆包/魔搭格式填写如 1024x1024。Gemini格式填写宽高比如 16:9 或 16:9-2K"),
            "seed": ConfigField(type=int, default=-1, description="随机种子，固定值可确保结果可复现"),
            "guidance_scale": ConfigField(type=float, default=2.5, description="指导强度。豆包推荐5.5，其他推荐2.5"),
            "num_inference_steps": ConfigField(type=int, default=30, description="推理步数，影响质量和速度。推荐20-50"),
            "watermark": ConfigField(type=bool, default=False, description="是否添加水印"),
            "custom_prompt_add": ConfigField(type=str, default="esthetic, excellent, medium resolution, newest", description="正面提示词增强，自动添加到用户描述后", input_type="textarea"),
            "negative_prompt_add": ConfigField(type=str, default="low quality, worst quality, bad quality, lowres, blurry, text, watermark, signature, extra arms, extra legs, extra hands, extra fingers, extra toes, missing fingers, bad anatomy, bad hands, bad proportions, extra thighs, extra calves, leg duplication, leg artifacts, logo, bubble, extra limbs", description="负面提示词，避免不良内容", input_type="textarea"),
            "artist": ConfigField(type=str, default="", description="艺术家风格标签（砂糖云专用）"),
            "support_img2img": ConfigField(type=bool, default=True, description="该模型是否支持图生图功能"),
            "auto_recall_delay": ConfigField(type=int, default=0, description="自动撤回延时（秒）"),
        },
        "models.model4": {
            "name": ConfigField(type=str, default="Sawata/Qwen-image-2512-Anime", description="模型显示名称，在模型列表中展示，版本更新后请手动从 old 目录恢复配置"),
            "base_url": ConfigField(type=str, default="https://api-inference.modelscope.cn/v1", description="API服务地址。示例: OpenAI=https://api.openai.com/v1, 硅基流动=https://api.siliconflow.cn/v1, 豆包=https://ark.cn-beijing.volces.com/api/v3, 魔搭=https://api-inference.modelscope.cn/v1, Gemini=https://generativelanguage.googleapis.com"),
            "api_key": ConfigField(type=str, default="Bearer xxxxxxxxxxxxxxxxxxxxxx", description="API密钥。OpenAI/modelscope格式需'Bearer '前缀，豆包/Gemini格式无需前缀", input_type="password", placeholder="Bearer sk-xxx 或 sk-xxx"),
            "format": ConfigField(type=str, default="modelscope", description="API格式。openai=通用格式，openai-chat=Chat接口生图(反重力反代Gemini等)，doubao=豆包，gemini=Gemini，modelscope=魔搭，shatangyun=砂糖云(NovelAI)，mengyuai=梦羽AI，zai=Zai(Gemini转发)", choices=["openai", "openai-chat", "gemini", "doubao", "modelscope", "shatangyun", "mengyuai", "zai"]),
            "model": ConfigField(type=str, default="Sawata/Qwen-image-2512-Anime", description="模型名称。梦羽AI格式填写模型索引数字（如0、1、2）"),
            "fixed_size_enabled": ConfigField(type=bool, default=False, description="是否固定图片尺寸。开启后强制使用default_size，关闭则麦麦选择"),
            "default_size": ConfigField(type=str, default="832x1216", description="默认图片尺寸。OpenAI/豆包/魔搭格式填写如 1024x1024。Gemini格式填写宽高比如 16:9 或 16:9-2K"),
            "seed": ConfigField(type=int, default=-1, description="随机种子，固定值可确保结果可复现"),
            "guidance_scale": ConfigField(type=float, default=2.5, description="指导强度。豆包推荐5.5，其他推荐2.5"),
            "num_inference_steps": ConfigField(type=int, default=30, description="推理步数，影响质量和速度。推荐20-50"),
            "watermark": ConfigField(type=bool, default=False, description="是否添加水印"),
            "custom_prompt_add": ConfigField(type=str, default="", description="正面提示词增强，自动添加到用户描述后", input_type="textarea"),
            "negative_prompt_add": ConfigField(type=str, default="low quality, worst quality, bad quality, lowres, blurry, text, watermark, signature, extra arms, extra legs, extra hands, extra fingers, extra toes, missing fingers, bad anatomy, bad hands, bad proportions, extra thighs, extra calves, leg duplication, leg artifacts, logo, bubble, extra limbs", description="负面提示词，避免不良内容", input_type="textarea"),
            "artist": ConfigField(type=str, default="", description="艺术家风格标签（砂糖云专用）"),
            "support_img2img": ConfigField(type=bool, default=True, description="该模型是否支持图生图功能"),
            "auto_recall_delay": ConfigField(type=int, default=0, description="自动撤回延时（秒）"),
        },
        "models.model5": {
            "name": ConfigField(type=str, default="cancel13/liaocao", description="模型显示名称，在模型列表中展示，版本更新后请手动从 old 目录恢复配置"),
            "base_url": ConfigField(type=str, default="https://api-inference.modelscope.cn/v1", description="API服务地址。示例: OpenAI=https://api.openai.com/v1, 硅基流动=https://api.siliconflow.cn/v1, 豆包=https://ark.cn-beijing.volces.com/api/v3, 魔搭=https://api-inference.modelscope.cn/v1, Gemini=https://generativelanguage.googleapis.com"),
            "api_key": ConfigField(type=str, default="Bearer xxxxxxxxxxxxxxxxxxxxxx", description="API密钥。OpenAI/modelscope格式需'Bearer '前缀，豆包/Gemini格式无需前缀", input_type="password", placeholder="Bearer sk-xxx 或 sk-xxx"),
            "format": ConfigField(type=str, default="modelscope", description="API格式。openai=通用格式，openai-chat=Chat接口生图(反重力反代Gemini等)，doubao=豆包，gemini=Gemini，modelscope=魔搭，shatangyun=砂糖云(NovelAI)，mengyuai=梦羽AI，zai=Zai(Gemini转发)", choices=["openai", "openai-chat", "gemini", "doubao", "modelscope", "shatangyun", "mengyuai", "zai"]),
            "model": ConfigField(type=str, default="cancel13/liaocao", description="模型名称。梦羽AI格式填写模型索引数字（如0、1、2）"),
            "fixed_size_enabled": ConfigField(type=bool, default=False, description="是否固定图片尺寸。开启后强制使用default_size，关闭则麦麦选择"),
            "default_size": ConfigField(type=str, default="832x1216", description="默认图片尺寸。OpenAI/豆包/魔搭格式填写如 1024x1024。Gemini格式填写宽高比如 16:9 或 16:9-2K"),
            "seed": ConfigField(type=int, default=-1, description="随机种子，固定值可确保结果可复现"),
            "guidance_scale": ConfigField(type=float, default=2.5, description="指导强度。豆包推荐5.5，其他推荐2.5"),
            "num_inference_steps": ConfigField(type=int, default=30, description="推理步数，影响质量和速度。推荐20-50"),
            "watermark": ConfigField(type=bool, default=False, description="是否添加水印"),
            "custom_prompt_add": ConfigField(type=str, default="", description="正面提示词增强，自动添加到用户描述后", input_type="textarea"),
            "negative_prompt_add": ConfigField(type=str, default="low quality, worst quality, bad quality, lowres, blurry, text, watermark, signature, extra arms, extra legs, extra hands, extra fingers, extra toes, missing fingers, bad anatomy, bad hands, bad proportions, extra thighs, extra calves, leg duplication, leg artifacts", description="负面提示词，避免不良内容", input_type="textarea"),
            "artist": ConfigField(type=str, default="", description="艺术家风格标签（砂糖云专用）"),
            "support_img2img": ConfigField(type=bool, default=True, description="该模型是否支持图生图功能"),
            "auto_recall_delay": ConfigField(type=int, default=0, description="自动撤回延时（秒）"),
        },
        "models.model6": {
            "name": ConfigField(type=str, default="Remile/Qwen-Image-2512-FusionLoRA-ByRemile", description="模型显示名称，在模型列表中展示，版本更新后请手动从 old 目录恢复配置"),
            "base_url": ConfigField(type=str, default="https://api-inference.modelscope.cn/v1", description="API服务地址。示例: OpenAI=https://api.openai.com/v1, 硅基流动=https://api.siliconflow.cn/v1, 豆包=https://ark.cn-beijing.volces.com/api/v3, 魔搭=https://api-inference.modelscope.cn/v1, Gemini=https://generativelanguage.googleapis.com"),
            "api_key": ConfigField(type=str, default="Bearer xxxxxxxxxxxxxxxxxxxxxx", description="API密钥。OpenAI/modelscope格式需'Bearer '前缀，豆包/Gemini格式无需前缀", input_type="password", placeholder="Bearer sk-xxx 或 sk-xxx"),
            "format": ConfigField(type=str, default="modelscope", description="API格式。openai=通用格式，openai-chat=Chat接口生图(反重力反代Gemini等)，doubao=豆包，gemini=Gemini，modelscope=魔搭，shatangyun=砂糖云(NovelAI)，mengyuai=梦羽AI，zai=Zai(Gemini转发)", choices=["openai", "openai-chat", "gemini", "doubao", "modelscope", "shatangyun", "mengyuai", "zai"]),
            "model": ConfigField(type=str, default="Remile/Qwen-Image-2512-FusionLoRA-ByRemile", description="模型名称。梦羽AI格式填写模型索引数字（如0、1、2）"),
            "fixed_size_enabled": ConfigField(type=bool, default=False, description="是否固定图片尺寸。开启后强制使用default_size，关闭则麦麦选择"),
            "default_size": ConfigField(type=str, default="832x1216", description="默认图片尺寸。OpenAI/豆包/魔搭格式填写如 1024x1024。Gemini格式填写宽高比如 16:9 或 16:9-2K"),
            "seed": ConfigField(type=int, default=-1, description="随机种子，固定值可确保结果可复现"),
            "guidance_scale": ConfigField(type=float, default=2.5, description="指导强度。豆包推荐5.5，其他推荐2.5"),
            "num_inference_steps": ConfigField(type=int, default=30, description="推理步数，影响质量和速度。推荐20-50"),
            "watermark": ConfigField(type=bool, default=False, description="是否添加水印"),
            "custom_prompt_add": ConfigField(type=str, default="", description="正面提示词增强，自动添加到用户描述后", input_type="textarea"),
            "negative_prompt_add": ConfigField(type=str, default="low quality, worst quality, bad quality, lowres, blurry, text, watermark, signature, extra arms, extra legs, extra hands, extra fingers, extra toes, missing fingers, bad anatomy, bad hands, bad proportions, extra thighs, extra calves, leg duplication, leg artifacts", description="负面提示词，避免不良内容", input_type="textarea"),
            "artist": ConfigField(type=str, default="", description="艺术家风格标签（砂糖云专用）"),
            "support_img2img": ConfigField(type=bool, default=True, description="该模型是否支持图生图功能"),
            "auto_recall_delay": ConfigField(type=int, default=0, description="自动撤回延时（秒）"),
        },
        "models.model7": {
            "name": ConfigField(type=str, default="Qwen/Qwen-Image-Edit-2511", description="模型显示名称，在模型列表中展示，版本更新后请手动从 old 目录恢复配置"),
            "base_url": ConfigField(type=str, default="https://api-inference.modelscope.cn/v1", description="API服务地址。示例: OpenAI=https://api.openai.com/v1, 硅基流动=https://api.siliconflow.cn/v1, 豆包=https://ark.cn-beijing.volces.com/api/v3, 魔搭=https://api-inference.modelscope.cn/v1, Gemini=https://generativelanguage.googleapis.com"),
            "api_key": ConfigField(type=str, default="Bearer xxxxxxxxxxxxxxxxxxxxxx", description="API密钥。OpenAI/modelscope格式需'Bearer '前缀，豆包/Gemini格式无需前缀", input_type="password", placeholder="Bearer sk-xxx 或 sk-xxx"),
            "format": ConfigField(type=str, default="modelscope", description="API格式。openai=通用格式，openai-chat=Chat接口生图(反重力反代Gemini等)，doubao=豆包，gemini=Gemini，modelscope=魔搭，shatangyun=砂糖云(NovelAI)，mengyuai=梦羽AI，zai=Zai(Gemini转发)", choices=["openai", "openai-chat", "gemini", "doubao", "modelscope", "shatangyun", "mengyuai", "zai"]),
            "model": ConfigField(type=str, default="Qwen/Qwen-Image-Edit-2511", description="模型名称。梦羽AI格式填写模型索引数字（如0、1、2）"),
            "fixed_size_enabled": ConfigField(type=bool, default=False, description="是否固定图片尺寸。开启后强制使用default_size，关闭则麦麦选择"),
            "default_size": ConfigField(type=str, default="1024x1024", description="默认图片尺寸。OpenAI/豆包/魔搭格式填写如 1024x1024。Gemini格式填写宽高比如 16:9 或 16:9-2K"),
            "seed": ConfigField(type=int, default=-1, description="随机种子，固定值可确保结果可复现"),
            "guidance_scale": ConfigField(type=float, default=2.5, description="指导强度。豆包推荐5.5，其他推荐2.5"),
            "num_inference_steps": ConfigField(type=int, default=30, description="推理步数，影响质量和速度。推荐20-50"),
            "watermark": ConfigField(type=bool, default=False, description="是否添加水印"),
            "custom_prompt_add": ConfigField(type=str, default="", description="正面提示词增强，自动添加到用户描述后", input_type="textarea"),
            "negative_prompt_add": ConfigField(type=str, default="low quality, worst quality, bad quality, lowres, blurry, text, watermark, signature, extra arms, extra legs, extra hands, extra fingers, extra toes, missing fingers, bad anatomy, bad hands, bad proportions, extra thighs, extra calves, leg duplication, leg artifacts, logo, bubble, extra limbs", description="负面提示词，避免不良内容", input_type="textarea"),
            "artist": ConfigField(type=str, default="", description="艺术家风格标签（砂糖云专用）"),
            "support_img2img": ConfigField(type=bool, default=True, description="该模型是否支持图生图功能"),
            "auto_recall_delay": ConfigField(type=int, default=0, description="自动撤回延时（秒）"),
        },
    }

    def __init__(self, plugin_dir: str):
        """初始化插件，集成增强配置管理器"""
        import toml
        # 在父类初始化前读取原始配置文件
        config_path = os.path.join(plugin_dir, self.config_file_name)
        original_config = None
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    original_config = toml.load(f)
                print(f"[CustomPicPlugin] 读取原始配置文件: {config_path}")
            except Exception as e:
                print(f"[CustomPicPlugin] 读取原始配置失败: {e}")
        
        # 先调用父类初始化，这会加载配置并可能触发 MaiBot 迁移
        super().__init__(plugin_dir)
        
        # 初始化增强配置管理器
        self.enhanced_config_manager = EnhancedConfigManager(plugin_dir, self.config_file_name)
        
        # 注入插件实例到 Command 类，以便 Command 可以访问配置管理器保存配置
        PicConfigCommand.plugin_instance = self
        
        # 检查并更新配置（如果需要），传入原始配置
        self._enhance_config_management(original_config)
        
        # 检查插件启用状态和定时自拍配置
        from src.common.logger import get_logger as get_logger_func
        plugin_logger = get_logger_func("custom_pic_plugin")
        
        plugin_enabled = self.get_config("plugin.enabled", False)
        auto_selfie_enabled = self.get_config("auto_selfie.enabled", False)
        
        # 注册定时任务
        if plugin_enabled and auto_selfie_enabled:
            from src.manager.async_task_manager import async_task_manager
            from .core.auto_selfie_task import AutoSelfieTask
            
            # 【关键修复】检查全局 abort_flag 状态并自动修复
            # 如果之前的操作（如停止所有任务）导致 abort_flag 仍处于 set 状态，
            # 新注册的任务将直接跳过执行。这里需要强制清除它。
            if async_task_manager.abort_flag.is_set():
                plugin_logger.warning("[CustomPicPlugin] 检测到全局任务中止标志异常，已自动重置")
                async_task_manager.abort_flag.clear()
            
            self._register_auto_selfie_task()
    
    def _register_auto_selfie_task(self):
        """注册定时自拍任务"""
        from src.common.logger import get_logger as get_logger_func
        plugin_logger = get_logger_func("custom_pic_plugin")
        
        try:
            from src.manager.async_task_manager import async_task_manager
            from .core.auto_selfie_task import AutoSelfieTask
            import asyncio

            # 创建并注册任务
            task = AutoSelfieTask(self)
            
            # 定义回调函数
            def _on_task_added(t):
                try:
                    t.result()
                    plugin_logger.info(f"[CustomPicPlugin] 定时自拍任务已成功注册")
                except Exception as e:
                    plugin_logger.error(f"[CustomPicPlugin] 定时自拍任务注册失败: {e}")

            # 提交给事件循环
            asyncio.create_task(async_task_manager.add_task(task, call_back=_on_task_added))
            
        except Exception as e:
            plugin_logger.error(f"[CustomPicPlugin] 注册定时任务时发生错误: {e}")
    
    def _enhance_config_management(self, original_config=None):
        """增强配置管理：备份、版本检查、智能合并
        
        Args:
            original_config: 从磁盘读取的原始配置（在父类初始化前读取），用于恢复用户自定义值
        """
        # 获取期望的配置版本
        expected_version = self._get_expected_config_version()
        
        # 将config_schema转换为EnhancedConfigManager需要的格式
        schema_for_manager = self._convert_schema_for_manager()
        
        # 生成默认配置结构
        default_config = self._generate_default_config_from_schema()
        
        # 确定要使用的旧配置：优先使用传入的原始配置，其次从备份文件加载
        old_config = original_config
        if old_config is None:
            old_dir = os.path.join(self.plugin_dir, "old")
            if os.path.exists(old_dir):
                import toml
                # 查找最新的备份文件（按时间戳排序），包括 auto_backup、new_backup 和 backup 文件
                backup_files = []
                for fname in os.listdir(old_dir):
                    if (fname.startswith(self.config_file_name + ".backup_") or
                        fname.startswith(self.config_file_name + ".new_backup_") or
                        fname.startswith(self.config_file_name + ".auto_backup_")) and fname.endswith(".toml"):
                        backup_files.append(fname)
                if backup_files:
                    # 按时间戳排序（文件名中包含 _YYYYMMDD_HHMMSS）
                    backup_files.sort(reverse=True)
                    latest_backup = os.path.join(old_dir, backup_files[0])
                    try:
                        with open(latest_backup, "r", encoding="utf-8") as f:
                            old_config = toml.load(f)
                        print(f"[CustomPicPlugin] 从备份文件加载原始配置: {backup_files[0]}")
                    except Exception as e:
                        print(f"[CustomPicPlugin] 加载备份文件失败: {e}")
        
        # 每次启动时创建备份（无论版本是否相同）
        # 加载当前配置文件以获取版本
        current_config = self.enhanced_config_manager.load_config()
        if current_config:
            current_version = self.enhanced_config_manager.get_config_version(current_config)
            print(f"[CustomPicPlugin] 当前配置版本 v{current_version}，创建启动备份")
            self.enhanced_config_manager.backup_config(current_version)
        else:
            print(f"[CustomPicPlugin] 配置文件不存在，跳过启动备份")
        
        # 使用增强配置管理器检查并更新配置
        # 传入旧配置（如果存在）以恢复用户自定义值
        updated_config = self.enhanced_config_manager.update_config_if_needed(
            expected_version=expected_version,
            default_config=default_config,
            schema=schema_for_manager,
            old_config=old_config
        )
        
        # 如果配置有更新，更新self.config
        if updated_config and updated_config != self.config:
            self.config = updated_config
            # 同时更新enable_plugin状态
            if "plugin" in self.config and "enabled" in self.config["plugin"]:
                self.enable_plugin = self.config["plugin"]["enabled"]
    
    def _get_expected_config_version(self) -> str:
        """获取期望的配置版本号"""
        if "plugin" in self.config_schema and isinstance(self.config_schema["plugin"], dict):
            config_version_field = self.config_schema["plugin"].get("config_version")
            if isinstance(config_version_field, ConfigField):
                return config_version_field.default
        return "1.0.0"
    
    def _convert_schema_for_manager(self) -> Dict[str, Any]:
        """将ConfigField格式的schema转换为EnhancedConfigManager需要的格式"""
        schema_for_manager = {}
        
        for section, fields in self.config_schema.items():
            if not isinstance(fields, dict):
                continue
                
            section_schema = {}
            for field_name, field in fields.items():
                if isinstance(field, ConfigField):
                    section_schema[field_name] = {
                        "description": field.description,
                        "default": field.default,
                        "required": field.required,
                        "choices": field.choices if field.choices else None,
                        "example": field.example
                    }
            
            schema_for_manager[section] = section_schema
        
        return schema_for_manager
    
    def _generate_default_config_from_schema(self) -> Dict[str, Any]:
        """从schema生成默认配置结构"""
        default_config = {}
        
        for section, fields in self.config_schema.items():
            if not isinstance(fields, dict):
                continue
                
            section_config = {}
            for field_name, field in fields.items():
                if isinstance(field, ConfigField):
                    section_config[field_name] = field.default
            
            default_config[section] = section_config
        
        return default_config

    def get_plugin_components(self) -> List[Tuple[ComponentInfo, Type]]:
        """返回插件包含的组件列表"""
        enable_unified_generation = self.get_config("components.enable_unified_generation", True)
        enable_pic_command = self.get_config("components.enable_pic_command", True)
        enable_pic_config = self.get_config("components.enable_pic_config", True)
        enable_pic_style = self.get_config("components.enable_pic_style", True)
        components = []

        if enable_unified_generation:
            components.append((CustomPicAction.get_action_info(), CustomPicAction))

        # 优先注册更具体的配置管理命令，避免被通用风格命令拦截
        if enable_pic_config:
            components.append((PicConfigCommand.get_command_info(), PicConfigCommand))

        if enable_pic_style:
            components.append((PicStyleCommand.get_command_info(), PicStyleCommand))

        # 最后注册通用的风格命令，以免覆盖特定命令
        if enable_pic_command:
            components.append((PicGenerationCommand.get_command_info(), PicGenerationCommand))

        return components
