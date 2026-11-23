# -*- coding: utf-8 -*-
"""
内存与显存监控模块
============================================================
后台守护线程按固定间隔采样，逐行写入运行目录下的
memory_monitor.csv（utf-8-sig，每行 flush + fsync），
即使进程被 OOM 杀掉（退出码 137），已写入的数据也保留，
事后可还原崩溃前的内存爬升曲线。

采样字段：
    wall_time          采样时刻（本地时间字符串）
    elapsed_s          监控启动后的秒数
    stage              业务阶段标签（入口/训练器随时设置）
    pid_rss_mb         本进程常驻物理内存 RSS
    ram_total/used/available_gb   系统内存总量/已用/可用
    swap_total/used_gb            swap 总量/已用
    gpu_alloc/reserved/peak_gb    显存已用/已预留/峰值
    gpu_total_gb                  显存总量

数据来源优先 psutil；未装 psutil 时回退到 /proc（Linux）
与 ctypes（Windows），不额外强制依赖。
"""
import csv
import os
import threading
import time

FIELDS = [
    "wall_time", "elapsed_s", "stage", "pid_rss_mb",
    "ram_total_gb", "ram_used_gb", "ram_available_gb",
    "swap_total_gb", "swap_used_gb",
    "gpu_alloc_gb", "gpu_reserved_gb", "gpu_peak_gb", "gpu_total_gb",
]

GB = 1024.0 ** 3
MB = 1024.0 ** 2


# ------------------------------------------------------------
# 系统信息后端：psutil 优先，失败回退 /proc 与 ctypes
# ------------------------------------------------------------
try:
    import psutil

    def _proc_rss_mb():
        return psutil.Process().memory_info().rss / MB

    def _system_memory():
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()
        return {
            "ram_total_gb": vm.total / GB,
            "ram_used_gb": vm.used / GB,
            "ram_available_gb": vm.available / GB,
            "swap_total_gb": sm.total / GB,
            "swap_used_gb": sm.used / GB,
        }
except ImportError:
    psutil = None

    def _read_meminfo():
        info = {}
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    key, rest = line.split(":", 1)
                    info[key] = float(rest.strip().split()[0]) * 1024.0
        except OSError:
            return None
        return info

    def _proc_rss_mb():
        # Linux：/proc/self/statm 第二字段为常驻页数
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            with open("/proc/self/statm", "r") as f:
                rss_pages = int(f.read().split()[1])
            return rss_pages * page_size / MB
        except OSError:
            return 0.0

    def _system_memory():
        info = _read_meminfo()
        if info is None:
            return dict.fromkeys(
                ("ram_total_gb", "ram_used_gb", "ram_available_gb",
                 "swap_total_gb", "swap_used_gb"), 0.0)
        total = info.get("MemTotal", 0.0)
        available = info.get("MemAvailable", info.get("MemFree", 0.0))
        return {
            "ram_total_gb": total / GB,
            "ram_used_gb": (total - available) / GB,
            "ram_available_gb": available / GB,
            "swap_total_gb": info.get("SwapTotal", 0.0) / GB,
            "swap_used_gb": info.get("SwapTotal", 0.0) / GB
                             - info.get("SwapFree", 0.0) / GB,
        }


def _gpu_memory():
    """GPU 显存读数；无 CUDA 时全部填 0。"""
    out = dict.fromkeys(
        ("gpu_alloc_gb", "gpu_reserved_gb", "gpu_peak_gb", "gpu_total_gb"), 0.0)
    try:
        import torch
        if not torch.cuda.is_available():
            return out
        out["gpu_alloc_gb"] = torch.cuda.memory_allocated() / GB
        out["gpu_reserved_gb"] = torch.cuda.memory_reserved() / GB
        out["gpu_peak_gb"] = torch.cuda.max_memory_allocated() / GB
        props = torch.cuda.get_device_properties(0)
        out["gpu_total_gb"] = props.total_memory / GB
    except Exception:
        pass
    return out


class MemoryMonitor:
    """后台监控器：守护线程定时采样并落盘，进程被杀时数据不丢。"""

    def __init__(self, csv_path, interval=5.0):
        self.csv_path = csv_path
        self.interval = float(interval)
        self._stage = "init"
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.t0 = time.time()
        self._fh = None
        self._writer = None

    def set_stage(self, label):
        """设置阶段标签，下一次采样即带上。"""
        with self._lock:
            self._stage = str(label)

    def _open_file(self):
        new_file = (not os.path.exists(self.csv_path)
                    or os.path.getsize(self.csv_path) == 0)
        # utf-8-sig 口径：仅在新建文件时写 BOM，追加不重复写
        self._fh = open(self.csv_path, "a", encoding="utf-8", newline="")
        if new_file:
            self._fh.write("\ufeff")
        self._writer = csv.DictWriter(self._fh, fieldnames=FIELDS)
        if new_file:
            self._writer.writeheader()
        self._fh.flush()

    def sample_once(self):
        """采集一行并立即落盘，崩溃前最后几秒的数据也能保留。"""
        with self._lock:
            stage = self._stage
        row = {
            "wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_s": round(time.time() - self.t0, 1),
            "stage": stage,
            "pid_rss_mb": round(_proc_rss_mb(), 2),
        }
        row.update(_system_memory())
        row.update(_gpu_memory())
        self._writer.writerow(row)
        # flush + fsync：被 SIGKILL 时缓冲区内数据不丢
        self._fh.flush()
        try:
            os.fsync(self._fh.fileno())
        except OSError:
            pass

    def _run(self):
        while not self._stop.is_set():
            try:
                self.sample_once()
            except Exception as exc:
                # 监控本身绝不能打断主流程
                print("内存监控采样失败：", exc)
            self._stop.wait(self.interval)

    def start(self):
        """启动后台监控线程。"""
        self._open_file()
        self._thread = threading.Thread(
            target=self._run, name="memory-monitor", daemon=True)
        self._thread.start()

    def stop(self):
        """停止线程并补采最后一行，关闭文件。"""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 2)
        try:
            self.sample_once()
        finally:
            if self._fh is not None:
                self._fh.close()
                self._fh = None


# ------------------------------------------------------------
# 模块级单例，业务代码直接调这几个函数
# ------------------------------------------------------------
_monitor = None


def start_monitor(csv_path, interval=5.0):
    """启动全局监控器；重复调用先停旧的再启动。"""
    global _monitor
    if _monitor is not None:
        stop_monitor()
    _monitor = MemoryMonitor(csv_path, interval=interval)
    _monitor.start()
    return _monitor


def set_stage(label):
    """设置当前阶段标签。"""
    if _monitor is not None:
        _monitor.set_stage(label)


def log_once(label=None):
    """立即补采一行（可临时指定标签），用于阶段切换点。"""
    if _monitor is None:
        return
    if label is not None:
        _monitor.set_stage(label)
    _monitor.sample_once()


def stop_monitor():
    """停止并关闭全局监控器。"""
    global _monitor
    if _monitor is not None:
        _monitor.stop()
        _monitor = None
