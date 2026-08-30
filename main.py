"""
RankAgent: Main Command-Line Interface.
"""
import sys
import argparse
import os
from orchestrator.state_machine import RankAgentOrchestrator

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # Windows terminals and redirected IDE consoles do not always share a code
    # page. Keep the single entrypoint UTF-8-safe without crashing on one symbol.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="RankAgent: Autonomous ML Research Agent for Recommender Systems")
    parser.add_argument("--data_dir", default=None, help="Path to KuaiRand-Pure dataset directory (optional, auto-detected)")
    parser.add_argument("--max_iterations", type=int, default=None, help="Iteration budget (default: configs/benchmark_kuairand.yaml, hard cap 50)")
    parser.add_argument("--max_wall_clock", type=int, default=None, help="Wall-clock ceiling in seconds (default: configs/benchmark_kuairand.yaml, 6h)")
    parser.add_argument("--skip_baseline", action="store_true",
                        help="Skip baseline reproduction (default: run the full pipeline)")
    args = parser.parse_args()

    print("==================================================================")
    print("         RankAgent: Autonomous RecSys ML Research Agent           ")
    print("==================================================================")
    
    agent = RankAgentOrchestrator(
        data_dir=args.data_dir,
        max_iterations=args.max_iterations,
        max_wall_clock=args.max_wall_clock,
        run_baseline=not args.skip_baseline
    )
    agent.start_loop()

if __name__ == "__main__":
    main()

