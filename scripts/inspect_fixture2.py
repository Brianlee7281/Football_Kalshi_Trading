#!/usr/bin/env python3
"""Inspect EPL fixture data structure in detail."""
import asyncio, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

from src.clients.goalserve import GoalserveClient
from src.clients.base_client import BaseClient

async def main():
    api_key = os.environ.get("GOALSERVE_API_KEY", "")
    async with GoalserveClient(api_key, timeout=60.0) as client:
        # Get EPL fixtures
        matches = await client.get_fixtures(1204)
        for m in matches:
            status = m.get("@status", m.get("status", ""))
            if str(status).upper() == "FT":
                # Print goals structure from fixture
                print("=== FIXTURE GOALS FIELD ===")
                print(json.dumps(m.get("goals"), indent=2)[:1000])
                print("\n=== FIXTURE HALFTIME ===")
                print(json.dumps(m.get("halftime"), indent=2)[:500])

                # Now fetch commentaries raw
                static_id = m.get("@static_id", "")
                print(f"\n=== RAW COMMENTARIES for {static_id} ===")
                http = BaseClient("https://www.goalserve.com/getfeed", timeout=60.0)
                resp = await http.get(
                    f"/{api_key}/commentaries/match",
                    params={"id": static_id, "league": "1204", "json": "1"},
                )
                data = resp.json()
                # Navigate to match
                match_data = data
                if "commentaries" in data:
                    if isinstance(data["commentaries"], dict):
                        t = data["commentaries"].get("tournament", {})
                        if isinstance(t, dict) and "match" in t:
                            match_data = t["match"]

                print(f"Match keys: {list(match_data.keys()) if isinstance(match_data, dict) else type(match_data)}")
                if isinstance(match_data, dict):
                    summary = match_data.get("summary")
                    print(f"\nsummary type: {type(summary)}")
                    print(f"summary value: {json.dumps(summary, indent=2)[:500] if summary else repr(summary)}")

                    # Check goals field
                    goals = match_data.get("goals")
                    print(f"\ngoals type: {type(goals)}")
                    print(f"goals value: {json.dumps(goals, indent=2)[:500] if goals else repr(goals)}")

                    # Check matchinfo for time
                    mi = match_data.get("matchinfo")
                    print(f"\nmatchinfo: {json.dumps(mi, indent=2)[:500] if mi else repr(mi)}")

                await http.close()
                break

asyncio.run(main())
