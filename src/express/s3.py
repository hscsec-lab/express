import os
import sys
import threading
import time
from collections import deque
from typing import Optional

class ProgressBase:
    """通用进度逻辑基类"""
    def __init__(self, use_ansi: bool = True, silent: bool = False):
        self._use_ansi = use_ansi
        self._silent = silent
        self._lock = threading.Lock()

        # 用于平滑速度计算的滑动窗口 (保存最近3秒的样本)
        self._samples = deque(maxlen=10)
        self._start_time = time.time()
        self._seen_so_far = 0

    def _format_bytes(self, num: float) -> str:
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if abs(num) < 1024.0:
                return f"{num:.1f} {unit}"
            num /= 1024.0
        return f"{num:.1f} PB"

    def _get_smooth_speed(self, current_total: int) -> float:
        now = time.time()

        # 1. 采样限流：避免高频调用撑死 deque
        if not self._samples or (now - self._samples[-1][0] >= 0.1):
            self._samples.append((now, current_total))

        if len(self._samples) < 2:
            return 0.0

        # 2. 计算窗口内的平均速度 (滑动平均)
        t1, b1 = self._samples[0]
        t2, b2 = self._samples[-1]

        dt = t2 - t1
        if dt <= 0: return 0.0

        return (b2 - b1) / dt

    def _write(self, s: str):
        if not self._silent:
            # \033[K 是清除光标到行尾的关键，彻底解决残留
            clear_line = "\033[K" if self._use_ansi else " " * 20
            sys.stdout.write(f"\r{s}{clear_line}")
            sys.stdout.flush()

class DownloadProgressSimple(ProgressBase):
    def __init__(self, filename: str = "", prefix: str = "Downloading", **kwargs):
        super().__init__(**kwargs)
        # 简化文件名显示
        self._display_name = (filename[:22] + '..') if len(filename) > 25 else filename
        self._prefix = prefix

    def __call__(self, bytes_amount: int):
        with self._lock:
            self._seen_so_far += bytes_amount
            speed = self._get_smooth_speed(self._seen_so_far)

            color = "\033[33m" if self._use_ansi else ""
            reset = "\033[0m" if self._use_ansi else ""

            msg = (f"{self._prefix}: {self._display_name} | "
                   f"{color}{self._format_bytes(self._seen_so_far)}{reset} "
                   f"[{self._format_bytes(speed)}/s]")
            self._write(msg)

    def done(self):
        self._write(f"{self._prefix}: {self._display_name} | \033[32mDone.\033[0m\n")


class ProgressPercentage(ProgressBase):
    def __init__(self, filename: str, prefix: str = "Progress", width: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self._filename = os.path.basename(filename)
        try:
            self._total_size = os.path.getsize(filename)
        except OSError:
            self._total_size = 0

        self._prefix = prefix
        self._bar_width = width or self._get_terminal_width()

    def _get_terminal_width(self):
        try:
            return min(max(os.get_terminal_size().columns - 50, 20), 40)
        except OSError:
            return 30

    def __call__(self, bytes_amount: int):
        with self._lock:
            self._seen_so_far += bytes_amount
            # 兼容上传可能超过总大小的特殊情况
            safe_seen = min(self._seen_so_far, self._total_size) if self._total_size > 0 else self._seen_so_far

            speed = self._get_smooth_speed(self._seen_so_far)
            ratio = (safe_seen / self._total_size) if self._total_size > 0 else 0

            # ETA 计算
            eta_str = "--:--"
            if ratio < 1 and speed > 0 and self._total_size > 0:
                remaining = (self._total_size - self._seen_so_far) / speed
                eta_str = time.strftime('%M:%S', time.gmtime(remaining)) if remaining < 3600 else " >1h"

            # 进度条组装
            filled = int(self._bar_width * ratio)
            bar = '█' * filled + '░' * (self._bar_width - filled)

            color = "\033[32m" if ratio >= 1 else "\033[33m"
            reset = "\033[0m" if self._use_ansi else ""

            line = (f"{self._prefix} |{bar}| {ratio:6.1%} "
                    f"{self._format_bytes(safe_seen)}/{self._format_bytes(self._total_size)} "
                    f"[{self._format_bytes(speed)}/s, ETA {eta_str}]")

            self._write(line)
            if ratio >= 1:
                sys.stdout.write("\n")