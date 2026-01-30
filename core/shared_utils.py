"""
共享工具模块

提供插件各模块共用的工具类和函数，避免代码重复。
"""

from typing import Optional, Any


class MockUserInfo:
    """模拟用户信息对象，用于构造虚拟消息"""

    def __init__(self, user_id: str, user_nickname: str, platform: str):
        self.user_id = user_id
        self.user_nickname = user_nickname
        self.platform = platform


class MockGroupInfo:
    """模拟群组信息对象，用于构造虚拟消息"""

    def __init__(self, group_id: str, group_name: str, group_platform: str):
        self.group_id = group_id
        self.group_name = group_name
        self.group_platform = group_platform


class MockChatInfo:
    """模拟聊天信息对象，用于构造虚拟消息"""

    def __init__(self, platform: str, group_info: Optional[MockGroupInfo] = None):
        self.platform = platform
        self.group_info = group_info


def extract_stream_info(chat_stream: Any) -> tuple:
    """从聊天流对象中提取信息

    Args:
        chat_stream: 聊天流对象

    Returns:
        tuple: (user_info, group_info, chat_info, is_group)
    """
    # 获取用户信息
    s_user_id = getattr(chat_stream.user_info, "user_id", "") if hasattr(chat_stream, "user_info") else ""
    s_user_nickname = (
        getattr(chat_stream.user_info, "user_nickname", "User") if hasattr(chat_stream, "user_info") else "User"
    )
    s_platform = getattr(chat_stream, "platform", "unknown")

    # 判断是否群聊
    is_group = getattr(chat_stream, "is_group", False)
    s_group_id = getattr(chat_stream, "group_id", "") if is_group else ""
    s_group_name = getattr(chat_stream, "group_name", "") if is_group else ""

    # 构造 Mock 对象
    user_info = MockUserInfo(s_user_id, s_user_nickname, s_platform)
    group_info = MockGroupInfo(s_group_id, s_group_name, s_platform) if is_group else None
    chat_info = MockChatInfo(s_platform, group_info)

    return user_info, group_info, chat_info, is_group


def create_mock_message(chat_stream: Any, message_id_prefix: str = "auto") -> Any:
    """创建模拟消息对象

    Args:
        chat_stream: 聊天流对象
        message_id_prefix: 消息ID前缀

    Returns:
        模拟的 DatabaseMessages 对象
    """
    import time
    from src.common.data_models.database_data_model import DatabaseMessages

    user_info, group_info, chat_info, _ = extract_stream_info(chat_stream)

    mock_message = DatabaseMessages()  # type: ignore
    mock_message.message_id = f"{message_id_prefix}_{int(time.time())}"
    mock_message.time = time.time()
    mock_message.user_info = user_info  # type: ignore[assignment]
    mock_message.chat_info = chat_info  # type: ignore[assignment]
    mock_message.processed_plain_text = f"{message_id_prefix} task"

    return mock_message
