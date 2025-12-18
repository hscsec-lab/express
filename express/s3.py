import os
import sys
import threading
import time
from typing import Optional


class DownloadProgressSimple:
    def __init__(
            self,
            filename: str = "",
            prefix: str = "Downloading",
            use_ansi: bool = True,
            silent: bool = False,
    ):
        self._filename = (filename[:25] + '..') if len(filename) > 27 else filename
        self._seen_so_far = 0
        self._start_time = time.time()
        self._last_time = self._start_time
        self._last_seen = 0
        self._lock = threading.Lock()
        self._prefix = prefix
        self._use_ansi = use_ansi
        self._silent = silent

        # Build label and record its length (for overwriting correctly)
        self._label = f"{self._prefix}: {self._filename}"
        try:
            max_label_len = min(max(os.get_terminal_size().columns - 30, 20), 40)
            self._label = self._label[:max_label_len]
        except OSError:
            pass  # keep as is
        self._label_len = len(self._label)

        if not self._silent:
            # Print label, then leave cursor at end (no \n)
            sys.stdout.write(self._label)
            sys.stdout.flush()

    def _format_bytes(self, num: float) -> str:
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if abs(num) < 1024.0:
                return f"{num:.1f} {unit}"
            num /= 1024.0
        return f"{num:.1f} PB"

    def __call__(self, bytes_amount: int):
        if bytes_amount <= 0 or self._silent:
            return

        with self._lock:
            self._seen_so_far += bytes_amount
            now = time.time()

            delta_bytes = self._seen_so_far - self._last_seen
            delta_time = now - self._last_time
            speed = delta_bytes / delta_time if delta_time > 0 else 0.0
            self._last_seen = self._seen_so_far
            self._last_time = now

            downloaded = self._format_bytes(self._seen_so_far)
            speed_str = f"{self._format_bytes(speed)}/s" if speed > 0 else "-- B/s"

            yellow = "\033[33m" if self._use_ansi else ""
            reset = "\033[0m" if self._use_ansi else ""

            # ✅ Key fix: \r + pad label area with spaces, then write progress
            # e.g.: "\rDownloading: model.bin          12.3 MB [1.2 MB/s]"
            progress_part = f"{yellow}{downloaded}{reset} [{speed_str}]"
            line = f"\r{self._label:<{self._label_len}} {progress_part}"

            sys.stdout.write(line)
            sys.stdout.flush()

    def done(self):
        """Finalize with green 'done' and newline."""
        if self._silent:
            return
        final_size = self._format_bytes(self._seen_so_far)
        green = "\033[32m" if self._use_ansi else ""
        reset = "\033[0m" if self._use_ansi else ""
        # Overwrite with final state + newline
        line = f"\r{self._label:<{self._label_len}} {green}{final_size}{reset} [done]\n"
        sys.stdout.write(line)
        sys.stdout.flush()

class ProgressPercentage:
    def __init__(
            self,
            filename: str,
            prefix: str = "Uploading",
            use_ansi: bool = True,
            width: Optional[int] = None,
            silent: bool = False
    ):
        self._filename = os.path.basename(filename)
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._start_time = time.time()
        self._last_time = self._start_time
        self._last_seen = 0
        self._lock = threading.Lock()
        self._prefix = prefix
        self._use_ansi = use_ansi
        self._silent = silent

        # Determine progress bar width
        if width is None:
            try:
                # Try to get terminal width; fallback to 50
                self._width = min(max(os.get_terminal_size().columns - 40, 20), 60)
            except OSError:  # e.g., in CI without TTY
                self._width = 50
        else:
            self._width = width

        self._bar_char = '█'
        self._empty_char = '░'

        if self._size <= 0:
            self._size = 1  # avoid div-by-zero; treat as unknown size

        if not self._silent:
            self._print_initial()

    def _format_bytes(self, num: float) -> str:
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if abs(num) < 1024.0:
                return f"{num:.1f} {unit}"
            num /= 1024.0
        return f"{num:.1f} PB"

    def _print_initial(self):
        name_display = (self._filename[:25] + '..') if len(self._filename) > 27 else self._filename
        action = f"{self._prefix}: {name_display}"
        size_str = self._format_bytes(self._size) if self._size > 0 else "???"
        sys.stdout.write(f"{action:<30} [{size_str:>8}]\n")
        sys.stdout.flush()

    def __call__(self, bytes_amount: int):
        if bytes_amount < 0:
            return  # ignore invalid calls

        with self._lock:
            self._seen_so_far += bytes_amount
            now = time.time()

            # Clamp seen to size to avoid >100% due to retries/over-fetch
            current = min(self._seen_so_far, self._size)

            # Calculate progress
            percentage = (current / self._size) * 100 if self._size > 0 else 0.0

            # Calculate speed (based on last chunk, smoother than avg)
            delta_bytes = current - self._last_seen
            delta_time = now - self._last_time
            speed = delta_bytes / delta_time if delta_time > 0 else 0.0
            self._last_seen = current
            self._last_time = now

            # ETA (only if progress > 0 and < 100%)
            eta = ""
            if 0 < percentage < 100 and speed > 0:
                remaining_bytes = self._size - current
                eta_sec = remaining_bytes / speed
                if eta_sec < 60:
                    eta = f" ETA {eta_sec:.0f}s"
                elif eta_sec < 3600:
                    eta = f" ETA {eta_sec / 60:.0f}m"
                else:
                    eta = f" ETA {eta_sec / 3600:.1f}h"

            # Build progress bar
            bar_len = int(self._width * percentage / 100)
            bar = self._bar_char * bar_len + self._empty_char * (self._width - bar_len)

            # ANSI colors (green for done, yellow for ongoing)
            green = "\033[32m" if self._use_ansi else ""
            yellow = "\033[33m" if self._use_ansi else ""
            reset = "\033[0m" if self._use_ansi else ""

            percent_str = f"{percentage:5.1f}%"
            speed_str = f"{self._format_bytes(speed)}/s" if speed > 0 else "-- B/s"
            current_str = self._format_bytes(current)
            total_str = self._format_bytes(self._size)

            status_color = green if percentage >= 100 else yellow
            line = (
                f"\r{status_color}{percent_str}{reset} "
                f"|{bar}| "
                f"{current_str}/{total_str} "
                f"[{speed_str}]{eta}"
            )

            if not self._silent:
                sys.stdout.write(line)
                sys.stdout.flush()

            # Final newline on completion
            if percentage >= 100:
                if not self._silent:
                    sys.stdout.write("\n")
                self._done = True

    # Optional: allow reuse
    def reset(self, filename: str):
        with self._lock:
            self.__init__(filename, self._prefix, self._use_ansi, self._width, self._silent)
