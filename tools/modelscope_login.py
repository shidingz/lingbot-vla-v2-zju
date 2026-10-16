#!/usr/bin/env python3
"""Log in to ModelScope without putting an access token in shell history."""

import argparse
from getpass import getpass
import logging
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="在终端隐藏输入魔搭 SDK Access Token，仅保存本机登录状态，不上传模型。"
    )
    parser.parse_args()
    if not sys.stdin.isatty():
        print("请在服务器的交互式终端运行本脚本。", file=sys.stderr)
        return 2

    os.umask(0o077)
    logging.disable(logging.CRITICAL)
    from modelscope.hub.api import HubApi, ModelScopeConfig

    try:
        token = getpass("粘贴魔搭 SDK Access Token（输入不回显），然后回车：").strip()
        if not token:
            print("没有输入令牌，已取消。", file=sys.stderr)
            return 2
        api = HubApi()
        _, cookies = api.login(token)
        del token
        if not cookies:
            print("未取得登录凭据，请检查令牌后重试。", file=sys.stderr)
            return 1
        username, _ = ModelScopeConfig.get_user_info()
        print(f"魔搭登录成功，账号：{username or '已认证'}。")
        print("现在可以通知助手继续上传；不要在聊天中发送令牌。")
        return 0
    except (KeyboardInterrupt, EOFError):
        print("\n已取消登录。", file=sys.stderr)
        return 130
    except Exception as exc:
        # HTTP exception details can contain sensitive request information.
        print(f"登录失败（{type(exc).__name__}）。请检查网络和令牌后重试。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
