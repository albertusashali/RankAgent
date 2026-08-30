"""
RankAgent: Main Command-Line Interface.
"""
import sys
import argparse
from orchestrator.state_machine import RankAgentOrchestrator

def main():
    parser = argparse.ArgumentParser(description="RankAgent: Autonomous ML Research Agent for Recommender Systems")
    parser.add_argument("--data_dir", default=None, help="Path to KuaiRand-Pure dataset directory (optional, auto-detected)")
    parser.add_argument("--max_iterations", type=int, default=None, help="Iteration budget (default: configs/benchmark_kuairand.yaml, hard cap 50)")
    parser.add_argument("--max_wall_clock", type=int, default=None, help="Wall-clock ceiling in seconds (default: configs/benchmark_kuairand.yaml, 6h)")
    # Baseline reproduction is ON by default: the challenge requires confirming
    # the official validation score before iterating, and the run log is the
    # evidence for it. --skip-baseline exists only for fast development loops and
    # marks the run as unverified in the log.
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Development shortcut: seed from the published 0.6016 instead of "
                             "reproducing it (~90s). The run is then marked unverified; "
                             "do not use it for a submitted run.")
    args = parser.parse_args()

    print("==================================================================")
    print("         RankAgent: Autonomous RecSys ML Research Agent           ")
    print("==================================================================")
    if args.skip_baseline:
        print("  WARNING: --skip-baseline set. The baseline will be asserted, not")
        print("           reproduced, and every delta will be relative to an")
        print("           unverified reference. Not valid for a submitted run.")
    
    agent = RankAgentOrchestrator(
        data_dir=args.data_dir,
        max_iterations=args.max_iterations,
        max_wall_clock=args.max_wall_clock,
        run_baseline=not args.skip_baseline,
    )
    agent.start_loop()

if __name__ == "__main__":
    main()

