"""local_runner 适配 shim：runner 总是以 `python -B <entry>` 启动 agent，
用 os.execv 原地替换为 rust-pro 二进制，stdin/stdout 协议透传。"""
import os
import pathlib

_here = pathlib.Path(__file__).resolve().parent
os.execv(str(_here / "target" / "release" / "rust-pro"), ["rust-pro"])
