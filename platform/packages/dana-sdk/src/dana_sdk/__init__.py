"""Dana AI Platform - Python SDK.

سازگار با OpenAI. فقط base_url و api_key را تغییر دهید.

Usage:
    from dana_sdk import Dana

    client = Dana(api_key="dk-...")
    response = client.chat("سلام!")
    print(response)
"""

from .client import Dana

__version__ = "0.1.0"
__all__ = ["Dana"]
