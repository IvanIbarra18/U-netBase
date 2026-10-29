
"""
Performance logger for PyTorch training and evaluation.

Metrics:
- Batch and epoch execution time
- Images per second (throughput)
- Process RAM usage
- CUDA allocated/reserved memory
- Peak CUDA memory usage
- CSV export

Optional dependency:
    pip install psutil
"""

import csv
import time
from datetime import datetime
from pathlib import Path

import torch

try:
    import psutil
except ImportError:
    psutil = None


class PerformanceLogger:
    def __init__(self,output_dir="performance_logs",device=None,experiment_name="unet",synchronize_cuda=True,synchronize_batches=True,save_csv=True,):
        """
        synchronize_cuda: synchronize CUDA at epoch boundaries for accurate
            epoch timings.
        synchronize_batches: also synchronize at every batch boundary. This
            gives exact per-batch GPU timings but stops the CPU from queuing
            the next batch while the GPU is busy, so disable it for
            production training runs.
        """
        self.output_dir = Path(output_dir)
        self.save_csv = save_csv
        if self.save_csv:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.experiment_name = experiment_name
        self.synchronize_cuda = synchronize_cuda
        self.synchronize_batches = synchronize_batches

        self.batch_csv = (self.output_dir / f"{experiment_name}_batches.csv")
        self.epoch_csv = (self.output_dir / f"{experiment_name}_epochs.csv")

        self.process = psutil.Process() if psutil else None

        self._epoch_start = None
        self._batch_start = None
        self._phase = None
        self._epoch = None

        self._epoch_images = 0
        self._epoch_batches = 0
        self._batch_times = []
        self._last_batch_end = None
        self._data_wait = 0.0
        self._epoch_data_wait = 0.0

    def _sync_cuda(self, batch_boundary=False):
        """Synchronize pending GPU work for more accurate timings."""
        if batch_boundary and not self.synchronize_batches:
            return
        if (
            self.synchronize_cuda
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        ):
            torch.cuda.synchronize(self.device)

    def _ram_mb(self):
        """Return this process's resident RAM usage in MiB."""
        if self.process is None:
            return None

        try:
            return self.process.memory_info().rss / (1024 ** 2)
        except (OSError, psutil.Error):
            return None

    def _gpu_memory(self):
        """Return current and peak CUDA memory in MiB."""
        if (
            self.device.type != "cuda"
            or not torch.cuda.is_available()
        ):
            return {
                "gpu_allocated_mb": None,
                "gpu_reserved_mb": None,
                "gpu_peak_allocated_mb": None,
                "gpu_peak_reserved_mb": None,
            }

        return {
            "gpu_allocated_mb": (torch.cuda.memory_allocated(self.device) / (1024 ** 2)),
            "gpu_reserved_mb": (torch.cuda.memory_reserved(self.device) / (1024 ** 2)),
            "gpu_peak_allocated_mb": (torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)),
            "gpu_peak_reserved_mb": (torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)),
        }

    @staticmethod
    def _append_csv(path, row):
        """Append one dictionary to a CSV file."""
        exists = path.exists() and path.stat().st_size > 0

        with path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(row.keys()))

            if not exists:
                writer.writeheader()

            writer.writerow(row)

    def start_epoch(self, epoch, phase="train"):
        """Start timing a training or validation phase."""
        if self._epoch_start is not None:
            raise RuntimeError("An epoch is already active.")

        self._sync_cuda()

        if (
            self.device.type == "cuda"
            and torch.cuda.is_available()
        ):
            torch.cuda.reset_peak_memory_stats(self.device)

        self._epoch = epoch
        self._phase = phase
        self._epoch_start = time.perf_counter()

        self._epoch_images = 0
        self._epoch_batches = 0
        self._batch_times = []
        self._last_batch_end = self._epoch_start
        self._epoch_data_wait = 0.0

    def start_batch(self, batch=None):
        """Start timing a batch.

        Call it right after the DataLoader yields the batch: the time since
        the previous end_batch() (or start_epoch()) is recorded as data wait,
        i.e. time spent blocked on loading instead of computing.
        """
        if self._epoch_start is None:
            raise RuntimeError("Call start_epoch() first.")

        if self._batch_start is not None:
            raise RuntimeError("A batch is already active.")

        self._data_wait = time.perf_counter() - self._last_batch_end
        self._epoch_data_wait += self._data_wait

        self._sync_cuda(batch_boundary=True)
        self._batch = batch
        self._batch_start = time.perf_counter()

    def end_batch(self, images):
        """Record timing and resource usage for a completed batch."""
        if self._batch_start is None:
            raise RuntimeError("Call start_batch() first.")

        if images < 0:
            raise ValueError("images must be non-negative.")

        self._sync_cuda(batch_boundary=True)

        elapsed = max(
            time.perf_counter() - self._batch_start,
            1e-12,
        )

        row = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "phase": self._phase,
            "epoch": self._epoch,
            "batch": self._batch,
            "images": images,
            "elapsed_seconds": elapsed,
            "data_wait_seconds": self._data_wait,
            "images_per_second": images / elapsed,
            "cpu_ram_mb": self._ram_mb(),
            **self._gpu_memory(),
        }

        if self.save_csv:
            self._append_csv(self.batch_csv, row)

        self._epoch_images += images
        self._epoch_batches += 1
        self._batch_times.append(elapsed)
        self._batch_start = None
        self._last_batch_end = time.perf_counter()

        return row

    def end_epoch(self):
        """Record the total duration and throughput of an epoch."""
        if self._epoch_start is None:
            raise RuntimeError("Call start_epoch() first.")

        if self._batch_start is not None:
            raise RuntimeError("Finish the active batch first.")

        self._sync_cuda()

        elapsed = max(time.perf_counter() - self._epoch_start,1e-12,)

        row = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "phase": self._phase,
            "epoch": self._epoch,
            "images": self._epoch_images,
            "batches": self._epoch_batches,
            "elapsed_seconds": elapsed,
            "images_per_second": self._epoch_images / elapsed,
            "data_wait_seconds": self._epoch_data_wait,
            "data_wait_fraction": self._epoch_data_wait / elapsed,
            "mean_batch_seconds": (sum(self._batch_times) / len(self._batch_times)
                if self._batch_times else None
            ),
            "cpu_ram_mb": self._ram_mb(),
            **self._gpu_memory(),
        }

        if self.save_csv:
            self._append_csv(self.epoch_csv, row)

        self._epoch_start = None
        self._phase = None
        self._epoch = None

        return row