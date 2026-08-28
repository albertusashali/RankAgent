"""
RankAgent: Main Command-Line Interface.
"""
import sys
import argparse
from orchestrator.state_machine import RankAgentOrchestrator

def main():
    parser = argparse.ArgumentParser(description="RankAgent: Autonomous ML Research Agent for Recommender Systems")
    parser.add_argument("--data_dir", default=None, help="Path to KuaiRand-Pure dataset directory (optional, auto-detected)")
    parser.add_argument("--max_iterations", type=int, default=50, help="Maximum number of research iterations (hard cap: 50)")
    parser.add_argument("--max_wall_clock", type=int, default=21600, help="Maximum wall-clock time in seconds (default: 21600 = 6h)")
    args = parser.parse_args()

    print("==================================================================")
    print("         RankAgent: Autonomous RecSys ML Research Agent           ")
    print("==================================================================")
    
    agent = RankAgentOrchestrator(
        data_dir=args.data_dir,
        max_iterations=args.max_iterations,
        max_wall_clock=args.max_wall_clock
    )
    agent.start_loop()

if __name__ == "__main__":
    main()

