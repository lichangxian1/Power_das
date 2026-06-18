"""权重输出目录解析: 让训练脚本把 .pth 存到和它日志同一目录。

优先级:
  1) 显式 --out_dir
  2) stdout 被重定向到日志文件时, 自动用该日志所在目录 (权重与 log 同目录)
  3) 否则回退 default (dataset/)
"""
import os


def resolve_out_dir(explicit=None, default="dataset"):
    if explicit:
        os.makedirs(explicit, exist_ok=True)
        return explicit
    # nohup ... > outputs/.../train.log 2>&1 时, fd1 指向该 log 文件
    try:
        tgt = os.readlink("/proc/self/fd/1")
    except OSError:
        tgt = ""
    if (
        tgt.startswith("/")
        and not tgt.startswith("/dev/")
        and not tgt.startswith("/proc/")
        and "pipe:" not in tgt
    ):
        d = os.path.dirname(tgt)
        if d and os.path.isdir(d):
            return d
    return default


def place(save_path, out_dir):
    """把 save_path 的目录换成 out_dir, 保留原文件名。"""
    return os.path.join(out_dir, os.path.basename(save_path))
