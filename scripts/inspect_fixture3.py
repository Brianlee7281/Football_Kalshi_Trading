#!/usr/bin/env python3
"""Check fixtures for redcards, yellowreds, and card-related fields."""
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
        matches = await client.get_fixtures(1204)
        # Find a match with red cards (search all matches)
        red_count = 0
        for m in matches:
            status = m.get("@status", m.get("status", ""))
            if str(status).upper() != "FT":
                continue
            # Check for cards-related keys
            for key in m.keys():
                if "card" in key.lower() or "red" in key.lower():
                    print(f"Found key '{key}': {json.dumps(m[key], indent=2)[:300]}")
                    red_count += 1

            # Also check a few match structures to find cards
            if red_count == 0:
                # Print all top-level keys of a completed match
                pass

        if red_count == 0:
            # Try first 3 completed matches - print all keys
            count = 0
            for m in matches:
                status = m.get("@status", m.get("status", ""))
                if str(status).upper() != "FT":
                    continue
                count += 1
                if count <= 3:
                    lt = m.get("localteam", {})
                    vt = m.get("visitorteam", {})
                    print(f"\n--- Match {m.get('@id')} ({lt.get('@name')} vs {vt.get('@name')}) ---")
                    print(f"Keys: {list(m.keys())}")
                    # Check goals for owngoal field
                    goals = m.get("goals", {})
                    if goals:
                        goal_list = goals.get("goal", [])
                        if isinstance(goal_list, dict):
                            goal_list = [goal_list]
                        if goal_list:
                            print(f"First goal keys: {list(goal_list[0].keys())}")
                            print(f"First goal: {json.dumps(goal_list[0], indent=2)}")

        # Search for red cards across all completed matches
        print("\n=== SEARCHING FOR RED CARDS IN ALL EPL FIXTURES ===")
        for m in matches:
            status = m.get("@status", m.get("status", ""))
            if str(status).upper() != "FT":
                continue
            # Check lineups for cards
            lineups = m.get("lineups", {})
            if isinstance(lineups, dict):
                for team in ("localteam", "visitorteam"):
                    team_data = lineups.get(team, {})
                    if isinstance(team_data, dict):
                        players = team_data.get("player", [])
                        if isinstance(players, dict):
                            players = [players]
                        if isinstance(players, list):
                            for p in players:
                                if p.get("@redcard") == "True" or p.get("@yellowred") == "True":
                                    lt = m.get("localteam", {})
                                    vt = m.get("visitorteam", {})
                                    print(f"RED CARD: {p.get('@name')} ({team}) in {lt.get('@name')} vs {vt.get('@name')} - minute: {p.get('@redcard_minute', p.get('@yellowred_minute', '?'))}")
                                    print(f"  Player data: {json.dumps(p, indent=2)[:300]}")

asyncio.run(main())
