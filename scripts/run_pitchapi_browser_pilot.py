"""Launch and close isolated browser verification servers for real-payload pilots."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", required=True)
    parser.add_argument("--playwright-module", required=True)
    parser.add_argument("--leagues", nargs="*", default=["belgium", "laliga", "ligue1", "eredivisie", "epl", "scotland", "turkey"])
    parser.add_argument("--consumers", action="store_true", help="Verify the real seven consumer entrypoints and navigation")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    for index, league in enumerate(args.leagues):
        repositories = {"belgium": "belgium-soccer", "laliga": "la-liga", "ligue1": "ligue-1", "eredivisie": "netherlands-soccer", "epl": "premier-league", "scotland": "scotland-premiership", "turkey": "turkey-soccer"}
        directory = root.parent / repositories[league] if args.consumers else root / "precomputed/pitchapi-ui" / league
        output = root / ("precomputed/pitchapi-consumer-browser" if args.consumers else "precomputed/pitchapi-browser") / league
        output.mkdir(parents=True, exist_ok=True)
        port = 8571 + index
        env = dict(os.environ, PLAYWRIGHT_MODULE=args.playwright_module)
        env.pop("PITCH_ORACLE_DATA_DIR", None)
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        with (output / "server.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "predictions.py" if args.consumers else "app.py", "--server.port", str(port), "--server.address", "127.0.0.1", "--server.headless", "true", "--browser.gatherUsageStats", "false"], cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
            try:
                for _ in range(100):
                    if process.poll() is not None:
                        raise RuntimeError(f"{league} server stopped; see {output / 'server.log'}")
                    try:
                        with urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=1) as response:
                            if response.status == 200:
                                break
                    except OSError:
                        time.sleep(.2)
                else:
                    raise TimeoutError(f"{league} server did not start")
                command = [args.node, str(root / "scripts/verify_pitchapi_browser.cjs"), f"http://127.0.0.1:{port}", str(output)]
                if args.consumers:
                    command.append("consumer")
                subprocess.run(command, env=env, check=True)
                print(f"{league}: browser passed", flush=True)
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)


if __name__ == "__main__":
    main()
