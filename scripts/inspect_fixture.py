#!/usr/bin/env python3
"""Inspect one EPL fixture to see available fields."""
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

async def main():
    api_key = os.environ.get("GOALSERVE_API_KEY", "")
    async with GoalserveClient(api_key, timeout=60.0) as client:
        # Get EPL fixtures
        matches = await client.get_fixtures(1204)
        # Find a completed match with score
        for m in matches:
            lt = m.get("localteam", {})
            vt = m.get("visitorteam", {})
            status = m.get("@status", m.get("status", ""))
            if str(status).upper() == "FT":
                print("=== COMPLETED MATCH ===")
                print(f"ID: {m.get('@id', m.get('id'))}")
                print(f"Static ID: {m.get('@static_id')}")
                print(f"Status: {status}")
                print(f"Local: {lt.get('@name', lt.get('name'))} - Score: {lt.get('@score', lt.get('score'))}")
                print(f"Visitor: {vt.get('@name', vt.get('name'))} - Score: {vt.get('@score', vt.get('score'))}")
                print(f"\nTop-level keys: {list(m.keys())}")
                print(f"\nSummary: {json.dumps(m.get('summary'), indent=2)[:500] if m.get('summary') else 'NONE'}")
                print(f"\nMatchinfo: {json.dumps(m.get('matchinfo'), indent=2)[:500] if m.get('matchinfo') else 'NONE'}")

                # Now fetch detailed stats for this match
                match_id = str(m.get("@static_id", m.get("@id", m.get("id", ""))))
                print(f"\n=== FETCHING COMMENTARIES for static_id={match_id} ===")
                try:
                    stats = await client.get_match_stats(match_id, 1204)
                    print(f"Stats keys: {list(stats.keys())}")
                    if stats.get("summary"):
                        print(f"Summary keys: {list(stats['summary'].keys()) if isinstance(stats['summary'], dict) else type(stats['summary'])}")
                        for team in ("localteam", "visitorteam"):
                            td = stats["summary"].get(team, {})
                            if td:
                                print(f"  {team} summary keys: {list(td.keys())}")
                                goals = td.get("goals", {})
                                if goals:
                                    print(f"    goals: {json.dumps(goals, indent=4)[:300]}")
                    else:
                        print("No summary in stats response")

                    # Check for red cards
                    if stats.get("summary"):
                        for team in ("localteam", "visitorteam"):
                            td = stats["summary"].get(team, {})
                            rc = td.get("redcards", {}) if td else {}
                            if rc:
                                print(f"  {team} redcards: {json.dumps(rc, indent=4)[:300]}")
                except Exception as e:
                    print(f"Failed: {e}")

                break

asyncio.run(main())
