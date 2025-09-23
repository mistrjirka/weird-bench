#!/usr/bin/env python3
import json, os, shutil, subprocess, sys, time, platform, re, argparse

REPO = "https://github.com/Saniel0/Reversan-Engine.git"
PROJECT_DIR = os.path.abspath("Reversan-Engine")
BUILD_DIR = os.path.join(PROJECT_DIR, "build")
OUTPUT_JSON = os.path.abspath("reversan_results.json")

def run(cmd, cwd=None, check=True, capture_output=True):
    return subprocess.run(cmd, cwd=cwd, check=check, capture_output=capture_output, text=True)

def find_gnu_time():
    for c in ["/usr/bin/time", shutil.which("time"), shutil.which("gtime")]:
        if not c: 
            continue
        try:
            p = subprocess.run([c, "-v", "true"], capture_output=True, text=True)
            if p.returncode == 0 and "Maximum resident set size" in (p.stderr or ""):
                return c
        except Exception:
            pass
    return None

GNU_TIME = find_gnu_time()

def clone_or_update():
    # Always delete and re-clone for clean state
    if os.path.isdir(PROJECT_DIR):
        print(f"Removing existing {PROJECT_DIR}")
        shutil.rmtree(PROJECT_DIR)
    
    print(f"Cloning {REPO} -> {PROJECT_DIR}")
    run(["git", "clone", "--depth", "1", REPO, PROJECT_DIR])

def cmake_build():
    os.makedirs(BUILD_DIR, exist_ok=True)
    print("Configuring CMake…")
    config_start = time.perf_counter()
    run(["cmake", "-S", ".", "-B", "build", "-DCMAKE_BUILD_TYPE=Release"], cwd=PROJECT_DIR)
    config_time = time.perf_counter() - config_start
    
    print("Building…")
    build_start = time.perf_counter()
    run(["cmake", "--build", "build", "-j"], cwd=PROJECT_DIR)
    build_time = time.perf_counter() - build_start
    
    total_compile_time = config_time + build_time
    print(f"Compile time: {total_compile_time:.2f}s (config: {config_time:.2f}s, build: {build_time:.2f}s)")
    
    return total_compile_time

def pick_binary():
    # binaries live in repo root after build
    cand = [
        os.path.join(PROJECT_DIR, "reversan_avx2"),
        os.path.join(PROJECT_DIR, "reversan_nosimd"),
        # anything named 'reversan*' as last resort
    ]
    for c in cand:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    for name in os.listdir(PROJECT_DIR):
        if name.startswith("reversan") and os.access(os.path.join(PROJECT_DIR, name), os.X_OK):
            return os.path.join(PROJECT_DIR, name)
    raise RuntimeError("No 'reversan*' binary found in repo root.")

def reversan_supports_threads(bin_path):
    try:
        p = run([bin_path, "--help"], check=False)
        txt = (p.stdout or "") + (p.stderr or "")
        return bool(re.search(r"(^|\s)-t(\s|=|,|/)|threads", txt, re.IGNORECASE))
    except Exception:
        return False

def parse_gnu_time(stderr_text):
    def get(rex, cast=float, default=None):
        m = re.search(rex, stderr_text)
        if not m:
            return default
        try:
            return cast(m.group(1).strip())
        except Exception:
            return m.group(1).strip()
    max_rss_kb = get(r"Maximum resident set size \(kbytes\):\s+(\d+)", int)
    user_sec   = get(r"User time \(seconds\):\s+([0-9.]+)")
    sys_sec    = get(r"System time \(seconds\):\s+([0-9.]+)")
    elapsed    = get(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s+([0-9:]+)", str)
    def hms_to_s(s):
        if not isinstance(s, str): return None
        parts = s.split(":")
        if len(parts)==2: return int(parts[0])*60 + float(parts[1])
        if len(parts)==3: return int(parts[0])*3600 + int(parts[1])*60 + float(parts[2])
        return None
    return {
        "max_rss_kb": max_rss_kb,
        "user_seconds": user_sec,
        "sys_seconds": sys_sec,
        "elapsed_seconds": hms_to_s(elapsed),
    }

def measure(cmd, cwd=None):
    if GNU_TIME:
        p = subprocess.run([GNU_TIME, "-v"] + cmd, cwd=cwd, capture_output=True, text=True)
        metrics = parse_gnu_time(p.stderr or "")
        return p.returncode, metrics, p.stdout, p.stderr
    start = time.perf_counter()
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    return p.returncode, {
        "max_rss_kb": None,
        "user_seconds": None,
        "sys_seconds": None,
        "elapsed_seconds": elapsed,
    }, p.stdout, p.stderr

def calculate_average_metrics(runs):
    """Calculate average metrics from multiple runs."""
    if not runs:
        return None
    
    # Collect all metrics
    elapsed_times = [r["metrics"]["elapsed_seconds"] for r in runs if r["metrics"]["elapsed_seconds"] is not None]
    max_rss_values = [r["metrics"]["max_rss_kb"] for r in runs if r["metrics"]["max_rss_kb"] is not None]
    user_times = [r["metrics"]["user_seconds"] for r in runs if r["metrics"]["user_seconds"] is not None]
    sys_times = [r["metrics"]["sys_seconds"] for r in runs if r["metrics"]["sys_seconds"] is not None]
    
    return {
        "elapsed_seconds": sum(elapsed_times) / len(elapsed_times) if elapsed_times else None,
        "max_rss_kb": sum(max_rss_values) / len(max_rss_values) if max_rss_values else None,
        "user_seconds": sum(user_times) / len(user_times) if user_times else None,
        "sys_seconds": sum(sys_times) / len(sys_times) if sys_times else None,
        "elapsed_seconds_min": min(elapsed_times) if elapsed_times else None,
        "elapsed_seconds_max": max(elapsed_times) if elapsed_times else None,
        "max_rss_kb_min": min(max_rss_values) if max_rss_values else None,
        "max_rss_kb_max": max(max_rss_values) if max_rss_values else None,
    }

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Reversan Engine with customizable test repetitions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--runs", "-r", 
        type=int, 
        default=1, 
        help="Number of times to run each test (default: 1)"
    )
    args = parser.parse_args()
    
    if args.runs < 1:
        print("Error: Number of runs must be at least 1", file=sys.stderr)
        sys.exit(1)
    
    results = {
        "meta": {
            "host": platform.node(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "gnu_time": bool(GNU_TIME),
            "repo": REPO,
            "test_runs_per_config": args.runs,
        },
        "build": {},
        "runs_depth": [],
        "runs_threads": [],
    }

    try:
        clone_or_update()
        compile_time = cmake_build()
        bin_path = pick_binary()
        results["build"]["binary"] = bin_path
        results["build"]["compile_time_seconds"] = compile_time
        results["build"]["time_measurer"] = "GNU time -v" if GNU_TIME else "fallback (wall-clock only)"

        # capture a bit of --help for audit
        help_out = subprocess.run([bin_path, "--help"], capture_output=True, text=True)
        results["build"]["help_snippet"] = ((help_out.stdout or "") + (help_out.stderr or ""))[:2000]

        threads_ok = reversan_supports_threads(bin_path)
        results["build"]["threads_supported"] = threads_ok

        # 1) depth sweep 1..12
        print(f"\n=== Running depth sweep tests (1-12) with {args.runs} run(s) each ===")
        for depth in range(1, 13):
            depth_runs = []
            for run_num in range(args.runs):
                run_label = f"run {run_num + 1}" if args.runs > 1 else ""
                print(f"Running depth {depth}{'...' if args.runs == 1 else f' {run_label}...'}")
                cmd = [bin_path, "--bot-vs-bot", "--depth", str(depth)]
                rc, metrics, out, err = measure(cmd)
                depth_runs.append({
                    "run": run_num + 1,
                    "returncode": rc,
                    "metrics": metrics,
                    "stderr_tail": (err or "")[-500:],
                })
            
            # Calculate average metrics if multiple runs
            if args.runs > 1:
                avg_metrics = calculate_average_metrics(depth_runs)
                results["runs_depth"].append({
                    "depth": depth,
                    "runs": depth_runs,
                    "average_metrics": avg_metrics,
                    "num_runs": args.runs,
                })
            else:
                # Single run - keep the original format for compatibility
                results["runs_depth"].append({
                    "depth": depth,
                    "returncode": depth_runs[0]["returncode"],
                    "metrics": depth_runs[0]["metrics"],
                    "stderr_tail": depth_runs[0]["stderr_tail"],
                })

        # 2) threads sweep @ depth 11
        print(f"\n=== Running threads sweep tests (1-8) at depth 11 with {args.runs} run(s) each ===")
        for t in range(1, 9):
            if not threads_ok:
                print(f"Skipping threads test {t} (threads not supported)")
                results["runs_threads"].append({
                    "threads": t, "skipped": True, "reason": "threads_not_supported_by_binary"
                })
                continue
            
            thread_runs = []
            for run_num in range(args.runs):
                run_label = f"run {run_num + 1}" if args.runs > 1 else ""
                print(f"Running with {t} thread(s){'...' if args.runs == 1 else f' {run_label}...'}")
                cmd = [bin_path, "--bot-vs-bot", "--depth", "11", "-t", str(t)]
                rc, metrics, out, err = measure(cmd)
                thread_runs.append({
                    "run": run_num + 1,
                    "returncode": rc,
                    "metrics": metrics,
                    "stderr_tail": (err or "")[-500:],
                })
            
            # Calculate average metrics if multiple runs
            if args.runs > 1:
                avg_metrics = calculate_average_metrics(thread_runs)
                results["runs_threads"].append({
                    "threads": t,
                    "runs": thread_runs,
                    "average_metrics": avg_metrics,
                    "num_runs": args.runs,
                })
            else:
                # Single run - keep the original format for compatibility
                results["runs_threads"].append({
                    "threads": t,
                    "returncode": thread_runs[0]["returncode"],
                    "metrics": thread_runs[0]["metrics"],
                    "stderr_tail": thread_runs[0]["stderr_tail"],
                })

        with open(OUTPUT_JSON, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nOK → results saved to {OUTPUT_JSON}")

    except subprocess.CalledProcessError as e:
        print("\nERROR (subprocess):", e, file=sys.stderr)
        if e.stdout: print("--- stdout ---\n", e.stdout, file=sys.stderr)
        if e.stderr: print("--- stderr ---\n", e.stderr, file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print("\nERROR:", e, file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
