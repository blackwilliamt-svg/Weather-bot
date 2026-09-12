"""Minimal Tkinter GUI: double-click friendly front end for the pipeline.

Three buttons — Run Test Pull, Run Full Pull, Show Inventory — each backed by
the same pipeline/inventory functions the CLI (__main__.py) uses. Runs are
executed on a background thread so the window stays responsive, with
progress and log messages marshalled back to the Tk main thread via a queue.
"""
from __future__ import annotations

import os
import sys

# When launched via pythonw.exe (no console window — the point of a
# double-click GUI), sys.stdout/sys.stderr are None. Anything in this
# process that tries to print/log to them (this module, or the pipeline's
# fallback logging) would crash without this guard.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from . import inventory, pipeline
from .config import DEFAULT_CONFIG, DEFAULT_FULL_STORE_PATH, DEFAULT_TEST_STORE_PATH

FULL_PULL_WARNING = (
    "This will fetch data for roughly 19,000-26,000 land grid points across "
    "North America, from 1940 to present.\n\n"
    "It can take several hours and download tens of gigabytes. Run this on a "
    "machine you can leave on for a while.\n\n"
    "Continue?"
)


class WeatherBotApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("WeatherBot — Data Ingestion")
        self.root.geometry("760x480")
        self.msg_queue: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self.running = False

        self._build_widgets()
        self.root.after(100, self._poll_queue)

    def _build_widgets(self) -> None:
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")

        self.test_btn = ttk.Button(top, text="Run Test Pull", command=self.run_test)
        self.full_btn = ttk.Button(top, text="Run Full Pull", command=self.run_full)
        self.inventory_btn = ttk.Button(top, text="Show Inventory", command=self.show_inventory)
        self.test_btn.pack(side="left", padx=(0, 8))
        self.full_btn.pack(side="left", padx=(0, 8))
        self.inventory_btn.pack(side="left")

        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(top, textvariable=self.status_var).pack(side="right")

        progress_frame = ttk.Frame(self.root, padding=(10, 0))
        progress_frame.pack(fill="x")
        self.progress = ttk.Progressbar(progress_frame, mode="determinate")
        self.progress.pack(fill="x")

        log_frame = ttk.Frame(self.root, padding=10)
        log_frame.pack(fill="both", expand=True)
        self.log_widget = scrolledtext.ScrolledText(log_frame, state="disabled", wrap="word")
        self.log_widget.pack(fill="both", expand=True)

    def _append_log(self, text: str) -> None:
        self.log_widget.configure(state="normal")
        self.log_widget.insert("end", text + "\n")
        self.log_widget.see("end")
        self.log_widget.configure(state="disabled")

    def _set_buttons_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.test_btn.configure(state=state)
        self.full_btn.configure(state=state)
        self.inventory_btn.configure(state=state)

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self.msg_queue.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                elif kind == "progress":
                    step, total = payload
                    self.progress["maximum"] = max(total, 1)
                    self.progress["value"] = step
                    self.status_var.set(f"Running... {step}/{total}")
                elif kind == "done":
                    self._on_run_finished(payload)
                elif kind == "error":
                    self._on_run_error(str(payload))
                elif kind == "idle":
                    self._set_buttons_enabled(True)
                    self.status_var.set("Idle")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    # --- run pulls -----------------------------------------------------

    def run_test(self) -> None:
        self._start_run(mode="test")

    def run_full(self) -> None:
        if not messagebox.askyesno("Run full pull?", FULL_PULL_WARNING):
            return
        self._start_run(mode="full")

    def _start_run(self, mode: str) -> None:
        if self.running:
            return
        self.running = True
        self._set_buttons_enabled(False)
        self.progress["value"] = 0
        self.status_var.set(f"Running {mode} pull...")
        self._append_log(f"--- Starting {mode} pull ---")
        threading.Thread(target=self._run_worker, args=(mode,), daemon=True).start()

    def _run_worker(self, mode: str) -> None:
        try:
            store_path = DEFAULT_TEST_STORE_PATH if mode == "test" else DEFAULT_FULL_STORE_PATH
            config = DEFAULT_CONFIG.with_overrides(store_path=store_path)

            def on_progress(step: int, total: int, message: str) -> None:
                self.msg_queue.put(("progress", (step, total)))
                self.msg_queue.put(("log", f"[{step}/{total}] {message}"))

            if mode == "test":
                summary = pipeline.run_test_mode(config, on_progress=on_progress)
            else:
                summary = pipeline.run_full_mode(config, on_progress=on_progress)

            failure_text = pipeline.format_failure_summary(summary)
            log_path = pipeline.write_failure_log(summary, config.store_path)
            self.msg_queue.put(("done", {
                "summary": summary, "failure_text": failure_text,
                "log_path": log_path, "store_path": config.store_path,
            }))
        except Exception as exc:  # noqa: BLE001 - surface any failure to the GUI, never crash silently
            self.msg_queue.put(("error", f"{type(exc).__name__}: {exc}"))

    def _on_run_finished(self, payload: dict) -> None:
        self.running = False
        self._set_buttons_enabled(True)
        self.status_var.set("Idle")
        summary = payload["summary"]
        self._append_log(f"--- Done: {summary['n_points']} points, {summary['n_batches']} batches ---")
        self._append_log(payload["failure_text"])
        if payload["log_path"]:
            self._append_log(f"Full failure list written to: {payload['log_path']}")
        messagebox.showinfo(
            "WeatherBot",
            f"Finished.\n{summary['n_points']} points, {summary['n_batches']} batches fetched.\n"
            f"Store: {payload['store_path']}\n\n{payload['failure_text'].splitlines()[0]}",
        )

    def _on_run_error(self, error_text: str) -> None:
        self.running = False
        self._set_buttons_enabled(True)
        self.status_var.set("Idle")
        self._append_log(f"ERROR: {error_text}")
        messagebox.showerror("WeatherBot", f"Run failed:\n{error_text}")

    # --- inventory -------------------------------------------------------

    def show_inventory(self) -> None:
        initial_dir = "data" if os.path.isdir("data") else "."
        path = filedialog.askdirectory(title="Select a Zarr store folder to inspect", initialdir=initial_dir)
        if not path:
            return
        self._append_log(f"--- Inspecting {path} ---")
        self.status_var.set("Inspecting store...")
        self._set_buttons_enabled(False)
        threading.Thread(target=self._inventory_worker, args=(path,), daemon=True).start()

    def _inventory_worker(self, path: str) -> None:
        try:
            text = inventory.summarize(path)
            self.msg_queue.put(("log", text))
        except Exception as exc:  # noqa: BLE001
            self.msg_queue.put(("error", f"Inventory failed: {exc}"))
        finally:
            self.msg_queue.put(("idle", None))


def main() -> None:
    root = tk.Tk()
    WeatherBotApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
