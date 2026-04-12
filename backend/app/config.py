"""
配置管理
统一从项目根目录的 .env 文件加载配置
"""

import os
from typing import Optional
from urllib.parse import urlparse
from dotenv import load_dotenv

# 加载项目根目录的 .env 文件
# 路径: MiroFish/.env (相对于 backend/app/config.py)
project_root_env = os.path.join(os.path.dirname(__file__), '../../.env')

if os.path.exists(project_root_env):
    load_dotenv(project_root_env, override=True)
else:
    # 如果根目录没有 .env，尝试加载环境变量（用于生产环境）
    load_dotenv(override=True)


class Config:
    """Flask配置类"""

    # Flask配置
    SECRET_KEY = os.environ.get('SECRET_KEY', 'mirofish-secret-key')
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'

    # JSON配置 - 禁用ASCII转义，让中文直接显示（而不是 \uXXXX 格式）
    JSON_AS_ASCII = False

    # LLM配置（统一使用OpenAI格式）
    LLM_LOCAL_MODE = os.environ.get('LLM_LOCAL_MODE', 'False').lower() == 'true'
    LLM_API_KEY = os.environ.get('LLM_API_KEY')
    LLM_BASE_URL = os.environ.get('LLM_BASE_URL', 'https://api.openai.com/v1')
    LLM_MODEL_NAME = os.environ.get('LLM_MODEL_NAME', 'gpt-4o-mini')
    # 本地模式下，OpenAI SDK依然要求传入非空api_key，这里使用占位值
    LLM_LOCAL_PLACEHOLDER_API_KEY = os.environ.get('LLM_LOCAL_PLACEHOLDER_API_KEY', 'local-no-key-required')

    # Zep配置
    ZEP_API_KEY = os.environ.get('ZEP_API_KEY')
    # 是否强制要求Zep（默认false，启用本地图谱替代）
    ZEP_REQUIRED = os.environ.get('ZEP_REQUIRED', 'False').lower() == 'true'

    # 文件上传配置
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '../uploads')
    ALLOWED_EXTENSIONS = {'pdf', 'md', 'txt', 'markdown'}

    # 文本处理配置
    DEFAULT_CHUNK_SIZE = 500  # 默认切块大小
    DEFAULT_CHUNK_OVERLAP = 50  # 默认重叠大小

    # OASIS模拟配置
    OASIS_DEFAULT_MAX_ROUNDS = int(os.environ.get('OASIS_DEFAULT_MAX_ROUNDS', '10'))
    OASIS_SIMULATION_DATA_DIR = os.path.join(os.path.dirname(__file__), '../uploads/simulations')

    # OASIS平台可用动作配置
    OASIS_TWITTER_ACTIONS = [
        'CREATE_POST', 'LIKE_POST', 'REPOST', 'FOLLOW', 'DO_NOTHING', 'QUOTE_POST'
    ]
    OASIS_REDDIT_ACTIONS = [
        'LIKE_POST', 'DISLIKE_POST', 'CREATE_POST', 'CREATE_COMMENT',
        'LIKE_COMMENT', 'DISLIKE_COMMENT', 'SEARCH_POSTS', 'SEARCH_USER',
        'TREND', 'REFRESH', 'DO_NOTHING', 'FOLLOW', 'MUTE'
    ]

    # Report Agent配置
    REPORT_AGENT_MAX_TOOL_CALLS = int(os.environ.get('REPORT_AGENT_MAX_TOOL_CALLS', '5'))
    REPORT_AGENT_MAX_REFLECTION_ROUNDS = int(os.environ.get('REPORT_AGENT_MAX_REFLECTION_ROUNDS', '2'))
    REPORT_AGENT_TEMPERATURE = float(os.environ.get('REPORT_AGENT_TEMPERATURE', '0.5'))

    @classmethod
    def is_local_llm_mode(cls, base_url: Optional[str] = None) -> bool:
        """是否启用本地LLM模式（无需真实API Key）"""
        if cls.LLM_LOCAL_MODE:
            return True

        url = (base_url or cls.LLM_BASE_URL or '').strip()
        if not url:
            return False

        try:
            hostname = (urlparse(url).hostname or '').lower()
        except Exception:
            hostname = ''

        local_hosts = {'localhost', '127.0.0.1', '0.0.0.0', 'host.docker.internal'}
        return hostname in local_hosts

    @classmethod
    def resolve_llm_api_key(
        cls,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> Optional[str]:
        """解析可用的LLM API Key；本地模式允许使用占位Key"""
        candidate = (api_key or cls.LLM_API_KEY or '').strip()
        if candidate:
            return candidate
        if cls.is_local_llm_mode(base_url):
            return cls.LLM_LOCAL_PLACEHOLDER_API_KEY
        return None

    @classmethod
    def resolve_zep_api_key(cls, api_key: Optional[str] = None) -> Optional[str]:
        """解析有效的Zep API Key，自动忽略示例占位值"""
        candidate = (api_key or cls.ZEP_API_KEY or '').strip()
        if not candidate:
            return None
        placeholder_values = {
            'your_zep_api_key_here',
            'your_zep_api_key',
            'replace_me',
        }
        if candidate.lower() in placeholder_values:
            return None
        return candidate
    
    @classmethod
    
    def validate(cls):
        """验证必要配置"""
        errors = []
        if not cls.resolve_llm_api_key(base_url=cls.LLM_BASE_URL):
            errors.append('LLM_API_KEY 未配置')
        if cls.ZEP_REQUIRED and not cls.resolve_zep_api_key():
            errors.append('ZEP_API_KEY 未配置（当前为严格模式）')
        return errors
