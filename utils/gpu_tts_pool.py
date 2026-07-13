import gc
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional, TypeVar

import torch

T = TypeVar("T")

_GPU_TTS_METHODS = frozenset({"VietTTS", "XTTS", "PiperTTS"})

_VRAM_PROFILES = {
    "VietTTS": {"reserved_mb": 1200, "per_job_mb": 400},
    "XTTS": {"reserved_mb": 3000, "per_job_mb": 1500},
    "PiperTTS": {"reserved_mb": 500, "per_job_mb": 250},
    "default": {"reserved_mb": 0, "per_job_mb": 256},
}


def cuda_free_bytes() -> int:
    if not torch.cuda.is_available():
        return 0
    free_b, _ = torch.cuda.mem_get_info()
    return int(free_b)


def release_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_cuda_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error" in msg


def vram_profile(t2s_method: str) -> tuple[int, int]:
    profile = _VRAM_PROFILES.get(t2s_method, _VRAM_PROFILES["default"])
    reserved_mb = int(os.getenv("TTS_VRAM_RESERVED_MB", profile["reserved_mb"]))
    per_job_mb = int(os.getenv("TTS_VRAM_PER_JOB_MB", profile["per_job_mb"]))
    return reserved_mb, per_job_mb


def compute_max_workers(t2s_method: str, task_count: int) -> int:
    cap = max(1, int(os.getenv("TTS_JOBS", "1")))
    if task_count <= 0:
        return 1
    if t2s_method not in _GPU_TTS_METHODS or not torch.cuda.is_available():
        return max(1, min(cap, task_count))

    reserved_mb, per_job_mb = vram_profile(t2s_method)
    free_b = cuda_free_bytes()
    usable_b = max(0, free_b - reserved_mb * 1024 * 1024)
    vram_workers = max(1, usable_b // (per_job_mb * 1024 * 1024))
    workers = max(1, min(cap, vram_workers, task_count))
    print(
        f"TTS VRAM:: free={free_b / 1e9:.2f}GB reserved={reserved_mb}MB "
        f"per_job={per_job_mb}MB -> workers={workers} (cap={cap})"
    )
    return workers


class GpuConcurrency:
    """Limits how many TTS worker threads run in parallel."""

    _semaphore: Optional[threading.Semaphore] = None
    _lock = threading.Lock()

    @classmethod
    def configure(cls, max_workers: int):
        with cls._lock:
            cls._semaphore = threading.Semaphore(max(1, max_workers))

    @classmethod
    def reset(cls):
        with cls._lock:
            cls._semaphore = None

    def __enter__(self):
        if GpuConcurrency._semaphore is not None:
            GpuConcurrency._semaphore.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        if GpuConcurrency._semaphore is not None:
            GpuConcurrency._semaphore.release()
        return False


class GpuJobSlot:
    """Waits for enough free VRAM before inference and releases CUDA memory after."""

    _depth = threading.local()

    def __init__(self, t2s_method: str):
        self.t2s_method = t2s_method
        self._per_job_bytes = vram_profile(t2s_method)[1] * 1024 * 1024
        self._use_gpu = t2s_method in _GPU_TTS_METHODS and torch.cuda.is_available()

    @classmethod
    def _current_depth(cls) -> int:
        return getattr(cls._depth, "value", 0)

    @classmethod
    def _set_depth(cls, value: int):
        cls._depth.value = value

    def __enter__(self):
        if not self._use_gpu:
            return self
        depth = GpuJobSlot._current_depth()
        if depth == 0:
            deadline = time.monotonic() + float(os.getenv("TTS_VRAM_WAIT_SEC", "60"))
            while cuda_free_bytes() < self._per_job_bytes:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"Timed out waiting for {self._per_job_bytes / 1e6:.0f}MB free CUDA memory"
                    )
                release_cuda()
                time.sleep(0.25)
        GpuJobSlot._set_depth(depth + 1)
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self._use_gpu:
            return False
        depth = GpuJobSlot._current_depth()
        GpuJobSlot._set_depth(max(0, depth - 1))
        if GpuJobSlot._current_depth() == 0:
            release_cuda()
        return False


def _run_with_retry(
    task,
    worker: Callable[..., T],
    t2s_method: str,
    max_retries: int,
    retry_delay: float,
) -> Optional[T]:
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            with GpuConcurrency():
                return worker(*task)
        except Exception as exc:
            last_error = exc
            release_cuda()
            if not is_cuda_oom(exc) or attempt >= max_retries:
                print(f"TTS task failed ({attempt + 1}/{max_retries + 1}): {exc}")
                return None
            wait = retry_delay * (attempt + 1)
            print(
                f"TTS CUDA OOM, retry {attempt + 1}/{max_retries} "
                f"after {wait:.1f}s (free={cuda_free_bytes() / 1e6:.0f}MB)"
            )
            time.sleep(wait)
    print(f"TTS task exhausted retries: {last_error}")
    return None


def run_parallel_with_retry(
    tasks: list,
    worker: Callable[..., T],
    t2s_method: str,
    max_workers: Optional[int] = None,
) -> list[Optional[T]]:
    if not tasks:
        return []

    max_workers = max_workers or compute_max_workers(t2s_method, len(tasks))
    max_retries = int(os.getenv("TTS_RETRIES", "3"))
    retry_delay = float(os.getenv("TTS_RETRY_DELAY_SEC", "2"))

    GpuConcurrency.configure(max_workers)
    print(f"TTS pool:: workers={max_workers} tasks={len(tasks)} retries={max_retries}")

    try:
        if max_workers <= 1:
            return [
                _run_with_retry(task, worker, t2s_method, max_retries, retry_delay)
                for task in tasks
            ]

        results: list[Optional[T]] = [None] * len(tasks)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _run_with_retry, task, worker, t2s_method, max_retries, retry_delay
                ): idx
                for idx, task in enumerate(tasks)
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return results
    finally:
        GpuConcurrency.reset()
        release_cuda()
