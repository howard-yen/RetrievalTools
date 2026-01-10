#!/usr/bin/env python3
"""Simple server monitor that periodically fetches stats from the retrieval server."""

import argparse
import time
import sys
from datetime import datetime

import requests


def clear_line():
    """Clear the current line in terminal."""
    sys.stdout.write("\033[2K\033[G")
    sys.stdout.flush()


def format_latency(ms: float) -> str:
    """Format latency value, converting to seconds if >= 10s."""
    if ms >= 10000:
        return f"{ms / 1000:.1f}s"
    return f"{ms:.1f}ms"


def print_stats(stats: dict, url: str):
    """Print stats in a clean format."""
    now = datetime.now().strftime("%H:%M:%S")
    
    active = stats["active_requests"]
    total = stats["total_requests"]
    search = stats["search"]
    visit = stats["visit"]
    latency = stats["latency"]
    
    # Color codes
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    DIM = "\033[2m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    # Color active requests based on load
    if active == 0:
        active_color = GREEN
    elif active <= 5:
        active_color = YELLOW
    else:
        active_color = RED
    
    # Color latency based on response time
    avg_latency = latency["recent_avg_ms"]
    if avg_latency < 100:
        latency_color = GREEN
    elif avg_latency < 500:
        latency_color = YELLOW
    else:
        latency_color = RED
    
    print(f"{BOLD}[{now}]{RESET} {url}")
    print(f"  Active: {active_color}{active}{RESET}  |  Total: {CYAN}{total}{RESET}")
    print(f"  Search: {search['active']} active / {search['total']} reqs / {CYAN}{search['queries_processed']}{RESET} queries")
    print(f"  Visit:  {visit['active']} active / {visit['total']} reqs / {CYAN}{visit['docs_processed']}{RESET} docs")
    avg_str = format_latency(avg_latency)
    max_str = format_latency(latency['max_ms'])
    min_str = format_latency(latency['min_ms'])
    print(f"  Latency: {latency_color}{avg_str}{RESET} avg {DIM}| {max_str} max | {min_str} min{RESET}")
    print("-" * 50)


def monitor(url: str, interval: float):
    """Main monitoring loop."""
    print(f"\n🔍 Monitoring server at {url}")
    print(f"   Refresh interval: {interval}s (Ctrl+C to stop)\n")
    print("=" * 50)
    
    last_stats = None
    busy_since = None
    
    while True:
        now = datetime.now()
        try:
            response = requests.get(f"{url}/stats", timeout=3)
            response.raise_for_status()
            stats = response.json()
            last_stats = stats
            busy_since = None
            print_stats(stats, url)
        except requests.exceptions.ConnectionError:
            print(f"[{now.strftime('%H:%M:%S')}] ❌ Connection failed - server down?")
            print("-" * 50)
            busy_since = None
        except requests.exceptions.Timeout:
            # Server is likely busy processing a request
            if busy_since is None:
                busy_since = now
            busy_duration = (now - busy_since).seconds
            
            YELLOW = "\033[93m"
            DIM = "\033[2m"
            RESET = "\033[0m"
            
            status = f"{YELLOW}🔄 Server busy{RESET} (processing request)"
            if busy_duration > 0:
                status += f" {DIM}[{busy_duration}s]{RESET}"
            
            # Show last known stats if available
            if last_stats:
                active = last_stats["active_requests"]
                total = last_stats["total_requests"]
                print(f"[{now.strftime('%H:%M:%S')}] {status}")
                print(f"  Last known: {active} active / {total} total requests")
            else:
                print(f"[{now.strftime('%H:%M:%S')}] {status}")
            print("-" * 50)
        except Exception as e:
            print(f"[{now.strftime('%H:%M:%S')}] ⚠️  Error: {e}")
            print("-" * 50)
            busy_since = None
        
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Monitor retrieval server stats")
    parser.add_argument("--url", default="http://localhost:8000", help="Server URL (default: http://localhost:8000)")
    parser.add_argument("--interval", "-i", type=float, default=2.0, help="Refresh interval in seconds (default: 2.0)")
    args = parser.parse_args()
    
    try:
        monitor(args.url, args.interval)
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()

