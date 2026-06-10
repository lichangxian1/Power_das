"""Utility for resolving output directories when running train_proxy as __main__."""
import os


def resolve_out_dir(out_dir):
    """Return out_dir if given, else 'dataset/'."""
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        return out_dir
    return "dataset"


def place(filename, out_dir):
    """Join out_dir and filename."""
    return os.path.join(out_dir, filename)
