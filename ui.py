#!/usr/bin/env python3
"""
Simple GUI to prepare Blender Benchmark and list detected hardware.

Flow:
- Show status bar at top.
- Step 1: Download Blender Benchmark CLI.
- Step 2: Download Blender version 4.5.0 via the launcher.
- List detected devices from `benchmark-launcher-cli devices -b 4.5.0`.
- Provide a "No GPU" checkbox auto-checked if no GPUs are detected.
"""

import threading
import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Dict
from types import SimpleNamespace

from benchmarks.blender import BlenderBenchmark
from unified_runner import UnifiedBenchmarkRunner
from unified_models import UnifiedBenchmarkResult


class BlenderDeviceUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Weird Bench - Hardware Detection")
        self.geometry("720x520")
        self.resizable(True, True)

        self.benchmark = BlenderBenchmark(output_dir="results")

        # UI elements
        self.status_var = tk.StringVar(value="Ready")
        self.no_gpu_var = tk.BooleanVar(value=False)

        # Top status bar
        self.status_frame = ttk.Frame(self)
        self.status_frame.pack(fill=tk.X, padx=10, pady=(10, 6))
        ttk.Label(self.status_frame, textvariable=self.status_var).pack(side=tk.LEFT)

        # Controls
        self.controls = ttk.Frame(self)
        self.controls.pack(fill=tk.X, padx=10, pady=6)
        self.btn_refresh = ttk.Button(self.controls, text="Refresh / Detect", command=self.refresh_async)
        self.btn_refresh.pack(side=tk.LEFT)
        # Single smart run button that auto-detects all vs selected
        self.btn_run = ttk.Button(self.controls, text="Run Benchmarks", command=self.run_benchmarks_async)
        self.btn_run.pack(side=tk.LEFT, padx=(10, 0))
        self.btn_quit = ttk.Button(self.controls, text="Quit", command=self.destroy)
        self.btn_quit.pack(side=tk.RIGHT)

        # Devices list
        self.devices_frame = ttk.LabelFrame(self, text="Available devices (Blender 4.5.0)")
        self.devices_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)
        columns = ("name", "framework")
        self.tree = ttk.Treeview(self.devices_frame, columns=columns, show="headings", height=14, selectmode="none")
        self.tree.heading("name", text="Device")
        self.tree.heading("framework", text="Framework")
        self.tree.column("name", width=480)
        self.tree.column("framework", width=120, anchor=tk.CENTER)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        # Prevent selection interactions
        self.tree.bind("<<TreeviewSelect>>", lambda e: self.tree.selection_remove(*self.tree.selection()))

        # No-GPU checkbox
        self.chk_frame = ttk.Frame(self)
        self.chk_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.chk_no_gpu = ttk.Checkbutton(self.chk_frame, text="No GPU (CPU-only)", variable=self.no_gpu_var)
        self.chk_no_gpu.pack(side=tk.LEFT)

        # Upload checkbox (auto-ticked)
        self.upload_var = tk.BooleanVar(value=True)
        self.chk_upload = ttk.Checkbutton(self.chk_frame, text="Upload results after run", variable=self.upload_var)
        self.chk_upload.pack(side=tk.LEFT, padx=(20, 0))

        # Benchmark selection
        self.sel_frame = ttk.LabelFrame(self, text="Select benchmarks to run")
        self.sel_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.sel_reversan = tk.BooleanVar(value=True)
        self.sel_llama = tk.BooleanVar(value=True)
        self.sel_7zip = tk.BooleanVar(value=True)
        self.sel_blender = tk.BooleanVar(value=True)

        ttk.Checkbutton(self.sel_frame, text="Reversan", variable=self.sel_reversan, command=self._on_selection_change).pack(side=tk.LEFT, padx=(8, 8))
        ttk.Checkbutton(self.sel_frame, text="Llama", variable=self.sel_llama, command=self._on_selection_change).pack(side=tk.LEFT, padx=(8, 8))
        ttk.Checkbutton(self.sel_frame, text="7zip", variable=self.sel_7zip, command=self._on_selection_change).pack(side=tk.LEFT, padx=(8, 8))
        ttk.Checkbutton(self.sel_frame, text="Blender", variable=self.sel_blender, command=self._on_selection_change).pack(side=tk.LEFT, padx=(8, 8))

        # Kick off initial detection
        self.after(200, self.refresh_async)
        # Disable run buttons until detection completes
        self._disable_run_buttons()
        # Initialize upload state based on selection
        self._on_selection_change()

    # --- Background operations ---
    def refresh_async(self):
        # Disable refresh button during work
        self.btn_refresh.config(state=tk.DISABLED)
        t = threading.Thread(target=self._refresh_workflow, daemon=True)
        t.start()

    def _refresh_workflow(self):
        try:
            # Step 1: Download Blender benchmark CLI
            self._set_status("Downloading Blender Benchmark CLI…")
            ok = self.benchmark.ensure_benchmark_downloaded()
            if not ok:
                self._notify_error("Failed to download Blender Benchmark CLI. See console for details.")
                return

            # Step 2: Download Blender 4.5.0 assets
            self._set_status("Downloading Blender 4.5.0 assets…")
            ok = self.benchmark.ensure_blender_downloaded("4.5.0")
            if not ok:
                self._notify_error("Failed to download Blender 4.5.0 via the launcher.")
                return

            # Step 3: List devices
            self._set_status("Detecting devices via Blender Benchmark…")
            devices = self.benchmark.list_devices("4.5.0")
            self._update_devices(devices)

            # Auto-tick No GPU when only CPU framework appears
            has_gpu = any(d.get("framework", "").upper() != "CPU" for d in devices)
            self.no_gpu_var.set(not has_gpu)

            # Enable run buttons after successful detection
            self._enable_run_buttons()
            
            self._set_status("Ready")
        finally:
            # Re-enable refresh button
            self.btn_refresh.config(state=tk.NORMAL)

    # --- UI helpers (must run in main thread) ---
    def _set_status(self, text: str):
        def _update():
            self.status_var.set(text)
        self.after(0, _update)

    def _notify_error(self, message: str):
        def _show():
            messagebox.showerror("Error", message)
            self.status_var.set("Error")
        self.after(0, _show)

    def _update_devices(self, devices: List[Dict[str, str]]):
        def _apply():
            # Clear existing
            for item in self.tree.get_children():
                self.tree.delete(item)
            # Insert rows
            for d in devices:
                name = d.get("name", "?")
                framework = d.get("framework", "?")
                self.tree.insert("", tk.END, values=(name, framework))
        self.after(0, _apply)

    # --- Run button state (locked until detection completes) ---
    def _set_run_button_state(self, state: str):
        def _apply():
            self.btn_run.config(state=state)
        self.after(0, _apply)

    def _disable_run_buttons(self):
        self._set_run_button_state(tk.DISABLED)

    def _enable_run_buttons(self):
        self._set_run_button_state(tk.NORMAL)

    def _on_selection_change(self):
        """Called when benchmark selection changes - auto-disable upload if partial selection."""
        all_selected = (self.sel_reversan.get() and self.sel_llama.get() and 
                       self.sel_7zip.get() and self.sel_blender.get())
        
        # Disable upload checkbox if not all benchmarks are selected
        if all_selected:
            self.chk_upload.config(state=tk.NORMAL)
        else:
            self.chk_upload.config(state=tk.DISABLED)
            self.upload_var.set(False)

    # --- Run benchmarks ---
    def _disable_run_controls(self):
        self.btn_run.config(state=tk.DISABLED)
        self.btn_refresh.config(state=tk.DISABLED)

    def _enable_run_controls(self):
        self.btn_run.config(state=tk.NORMAL)
        self.btn_refresh.config(state=tk.NORMAL)

    def run_benchmarks_async(self):
        """Smart run function - detects all vs selected automatically."""
        self._disable_run_controls()
        
        # Check if all benchmarks are selected
        all_selected = (self.sel_reversan.get() and self.sel_llama.get() and 
                       self.sel_7zip.get() and self.sel_blender.get())
        
        if all_selected:
            t = threading.Thread(target=self._run_all_workflow, daemon=True)
        else:
            t = threading.Thread(target=self._run_selected_workflow, daemon=True)
        
        t.start()

    def _run_all_workflow(self):
        try:
            no_gpu = bool(self.no_gpu_var.get())
            upload = bool(self.upload_var.get())
            self._set_status("Preparing to run all benchmarks…")

            runner = UnifiedBenchmarkRunner(output_dir="results")
            runner.detect_hardware(cpu_only=no_gpu)

            args = SimpleNamespace(
                benchmark="all",
                output_dir="results",
                format="json",
                no_gpu=no_gpu,
                skip_build=False,
                gpu_device=None,
                vk_driver=None,
                api_url="https://weirdbench.eu/api",
                upload=upload,
            )

            self._set_status("Running all benchmarks… this may take a while")
            result = runner.run_all_benchmarks(args)
            self._set_status("Saving results…")
            filepath = runner.save_results(result, args.format)

            if upload:
                self._set_status("Uploading results…")
                runner.upload_results(result)

            self._set_status("Completed")
            self.after(0, lambda: messagebox.showinfo("Benchmarks completed", f"Results saved to:\n{filepath}"))
        except Exception as e:
            self._notify_error(f"Benchmark run failed: {e}")
        finally:
            self._enable_run_controls()



    def _run_selected_workflow(self):
        try:
            selected = []
            if self.sel_reversan.get():
                selected.append("reversan")
            if self.sel_llama.get():
                selected.append("llama")
            if self.sel_7zip.get():
                selected.append("7zip")
            if self.sel_blender.get():
                selected.append("blender")

            if not selected:
                self._notify_error("Please select at least one benchmark to run.")
                return

            no_gpu = bool(self.no_gpu_var.get())
            upload = bool(self.upload_var.get())

            self._set_status("Preparing to run selected benchmarks…")
            runner = UnifiedBenchmarkRunner(output_dir="results")
            runner.detect_hardware(cpu_only=no_gpu)

            # args namespace for per-benchmark calls
            args = SimpleNamespace(
                output_dir="results",
                format="json",
                no_gpu=no_gpu,
                skip_build=False,
                gpu_device=None,
                vk_driver=None,
                api_url="https://weirdbench.eu/api",
            )

            unified = UnifiedBenchmarkResult(meta=runner.system_info)

            for name in selected:
                self._set_status(f"Running {name} benchmark…")
                try:
                    if name == "reversan":
                        unified.reversan = runner._run_reversan_benchmark(args)
                    elif name == "llama":
                        unified.llama = runner._run_llama_benchmark(args)
                    elif name == "7zip":
                        unified.sevenzip = runner._run_7zip_benchmark(args)
                    elif name == "blender":
                        unified.blender = runner._run_blender_benchmark(args)
                except Exception as e:
                    # Continue other benchmarks, but report error
                    self._notify_error(f"{name} failed: {e}")

            self._set_status("Saving results…")
            # Save via runner to keep naming consistent
            filepath = UnifiedBenchmarkRunner(output_dir="results").save_results(unified, "json")

            if upload:
                self._set_status("Uploading results…")
                UnifiedBenchmarkRunner(output_dir="results").upload_results(unified)

            self._set_status("Completed")
            self.after(0, lambda: messagebox.showinfo("Benchmarks completed", f"Results saved to:\n{filepath}"))
        except Exception as e:
            self._notify_error(f"Benchmark run failed: {e}")
        finally:
            self._enable_run_controls()


def main():
    app = BlenderDeviceUI()
    app.mainloop()


if __name__ == "__main__":
    main()
