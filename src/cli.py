#!/usr/bin/env python3
"""CoherenceBench CLI — run benchmarks, view results, analyze runs."""

from __future__ import annotations

import argparse
import http.server
import json
import logging
import socket
import sys
import threading
import webbrowser
from pathlib import Path

from dotenv import load_dotenv


def cmd_run(args):
    """Run a benchmark."""
    load_dotenv()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.providers import get_provider
    from src.runner import BenchmarkRunner
    from src.analyzer import ResponseAnalyzer
    from src.scenarios import get_scenario

    scenario_name = args.scenario
    scenario = get_scenario(scenario_name)

    provider_kwargs = {}
    if args.model:
        provider_kwargs["model"] = args.model
    provider = get_provider(args.provider, **provider_kwargs)

    output_dir = Path(f"results/run_a_baseline/{args.provider}/seed_{args.seed}")

    def progress(event):
        if event.get("event") == "tick_completed":
            t = event["tick_number"]
            total = event["total_ticks"]
            dur = event.get("duration_seconds", 0)
            gt = event.get("ground_truth", {})
            correct = gt.get("correct_action", "?")
            chosen = "?"
            resp = event.get("response", "")
            import re
            m = re.search(r"ACTION:\s*(\S+)", resp, re.IGNORECASE)
            if m:
                chosen = m.group(1).strip().lower().rstrip(".")
            hit = "hit" if chosen in [a.lower() for a in gt.get("acceptable_actions", [])] else "miss"
            print(f"Tick {t:3d}/{total} | {dur:.1f}s | {hit} | chose={chosen} expected={correct}", flush=True)
        elif event.get("event") == "run_started":
            print(f"Started: {event.get('provider_name')} on {scenario_name}", flush=True)
        elif event.get("event") == "run_completed":
            print(f"\nRun completed!", flush=True)

    print(f"Provider: {provider.name()}")
    print(f"Scenario: {scenario_name} | Ticks: {args.num_ticks} | Seed: {args.seed}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    runner = BenchmarkRunner(
        provider=provider,
        num_ticks=args.num_ticks,
        seed=args.seed,
        output_dir=output_dir,
        scenario=scenario,
        progress_callback=progress,
    )

    import time
    start = time.time()
    results = runner.run(resume=args.resume)
    elapsed = time.time() - start
    print(f"\nCompleted {len(results)} ticks in {elapsed:.1f}s ({elapsed/len(results):.1f}s/tick)")

    # Analyze
    analyzer = ResponseAnalyzer(scenario=scenario)
    analyses = analyzer.analyze_run(results)

    from src.metrics import decision_fade_gap
    da_values = [a.decision_accuracy for a in analyses]
    dfg_data = decision_fade_gap(da_values)

    print(f"\nDA: {sum(da_values)/len(da_values):.1%}")
    print(f"DA@40: {dfg_data['da_first_40']:.1%}")
    print(f"DA@last: {dfg_data['da_last_40']:.1%}")
    print(f"DFG: {dfg_data['dfg']:+.1%}")

    # Generate viewer
    from src.viewer.generate import generate_viewer_html
    viewer_path = generate_viewer_html(output_dir, scenario_name)
    print(f"\nViewer: {viewer_path}")


def cmd_view(args):
    """Generate and serve the replay viewer."""
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    raw_file = results_dir / "raw_results.jsonl"
    if not raw_file.exists():
        print(f"Error: No raw_results.jsonl in {results_dir}", file=sys.stderr)
        sys.exit(1)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.viewer.generate import generate_viewer_html

    scenario = args.scenario or "power_grid"
    viewer_path = generate_viewer_html(results_dir, scenario)
    print(f"Generated: {viewer_path}")

    # Find free port
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()

    # Serve
    serve_dir = str(viewer_path.parent)

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=serve_dir, **kw)
        def log_message(self, format, *a):
            pass  # silence logs

    server = http.server.HTTPServer(('', port), QuietHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    url = f"http://localhost:{port}/{viewer_path.name}"
    print(f"Serving at {url}")
    print("Press Ctrl+C to stop")
    webbrowser.open(url)

    try:
        thread.join()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.shutdown()


def cmd_analyze(args):
    """Analyze a benchmark run and print summary."""
    results_dir = Path(args.results_dir)
    raw_file = results_dir / "raw_results.jsonl"
    if not raw_file.exists():
        print(f"Error: No raw_results.jsonl in {results_dir}", file=sys.stderr)
        sys.exit(1)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.analyzer import ResponseAnalyzer
    from src.scenarios import get_scenario
    from src.metrics import decision_fade_gap

    scenario = get_scenario(args.scenario or "power_grid")
    analyzer = ResponseAnalyzer(scenario=scenario)

    results = []
    with open(raw_file) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    analyses = analyzer.analyze_run(results)
    da_values = [a.decision_accuracy for a in analyses]
    fc_values = [a.factor_coverage for a in analyses]
    dfg_data = decision_fade_gap(da_values)

    print(f"Ticks: {len(analyses)}")
    print(f"DA:       {sum(da_values)/len(da_values):.1%}")
    print(f"FC:       {sum(fc_values)/len(fc_values):.1%}")
    print(f"DA@40:    {dfg_data['da_first_40']:.1%}")
    print(f"DA@last:  {dfg_data['da_last_40']:.1%}")
    print(f"DFG:      {dfg_data['dfg']:+.1%}")


def main():
    parser = argparse.ArgumentParser(
        prog="coherencebench",
        description="CoherenceBench — Measuring Decision Coherence in Long-Running Agents",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = sub.add_parser("run", help="Run a benchmark")
    p_run.add_argument("--provider", required=True, help="Provider name (groq, ollama, openrouter, etc.)")
    p_run.add_argument("--scenario", default="power_grid", help="Scenario (power_grid, hospital, network, air_traffic_control)")
    p_run.add_argument("--seed", type=int, default=42, help="Random seed")
    p_run.add_argument("--num-ticks", type=int, default=200, help="Number of ticks")
    p_run.add_argument("--model", default=None, help="Model override")
    p_run.add_argument("--resume", action="store_true", help="Resume from checkpoint (existing raw_results.jsonl)")
    p_run.set_defaults(func=cmd_run)

    # view
    p_view = sub.add_parser("view", help="Open replay viewer for a run")
    p_view.add_argument("results_dir", help="Path to results directory")
    p_view.add_argument("--scenario", default=None, help="Scenario name for phase weights")
    p_view.set_defaults(func=cmd_view)

    # analyze
    p_analyze = sub.add_parser("analyze", help="Print summary metrics")
    p_analyze.add_argument("results_dir", help="Path to results directory")
    p_analyze.add_argument("--scenario", default=None, help="Scenario name")
    p_analyze.set_defaults(func=cmd_analyze)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
