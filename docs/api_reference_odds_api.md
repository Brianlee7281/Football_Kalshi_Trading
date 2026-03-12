# Odds-API.io Complete Documentation
# Full API documentation for AI tools with large context windows
# Last updated: December 2024

================================================================================
OVERVIEW
================================================================================

Odds-API.io is a sports betting odds comparison API providing real-time data from 265 bookmakers across 34 sports with near-zero latency.

Key Features:
- 265 bookmakers supported (242 currently active)
- 34 sports coverage
- Real-time odds with close to zero latency
- Pre-match & In-play odds coverage
- Value bet detection
- Arbitrage opportunity detection
- WebSocket real-time updates
- Direct bet links to bookmaker sites

Base URL: https://api.odds-api.io/v3

================================================================================
AUTHENTICATION
================================================================================

Odds-API.io uses API key authentication. Include your API key as a query parameter:

  https://api.odds-api.io/v3/events?apiKey=YOUR_API_KEY&sport=football

Getting Your API Key:
1. Sign up for free at https://odds-api.io
2. Check your email for the API key
3. Store it securely in environment variables

Security Best Practices:
- Never expose API keys in client-side code
- Use environment variables
- Make API calls from backend servers only

Rate Limits:
- 5,000 requests/hour on all plans
- Additional packages available: +10K/20K/30K requests/hour
- Rate limit headers returned with every response:
  - x-ratelimit-limit: Total requests allowed
  - x-ratelimit-remaining: Requests remaining
  - x-ratelimit-reset: ISO 8601 timestamp when limit resets

HTTP Status Codes:
- 200: Success
- 400: Bad Request - Invalid parameters
- 401: Unauthorized - Invalid or missing API key
- 403: Forbidden - API key doesn't have access
- 404: Not Found - Resource doesn't exist
- 429: Too Many Requests - Rate limit exceeded
- 500: Internal Server Error

================================================================================
API ENDPOINTS
================================================================================

------------------------------------------------------------------------------
GET /sports
------------------------------------------------------------------------------
Returns list of available sports. No authentication required.

Response:
[
  {"name": "Football", "slug": "football"},
  {"name": "Basketball", "slug": "basketball"},
  ...
]

------------------------------------------------------------------------------
GET /bookmakers
------------------------------------------------------------------------------
Returns list of available bookmakers. No authentication required.

Response:
[
  {"name": "Bet365", "active": true},
  {"name": "SingBet", "active": true},
  ...
]

------------------------------------------------------------------------------
GET /leagues
------------------------------------------------------------------------------
Returns leagues for a specified sport.

Parameters:
- apiKey (required): Your API key
- sport (required): Sport slug (e.g., "football")

Example: /leagues?apiKey=YOUR_KEY&sport=football

Response:
[
  {"name": "England - Premier League", "slug": "england-premier-league", "eventsCount": 32},
  {"name": "Spain - La Liga", "slug": "spain-laliga", "eventsCount": 28},
  ...
]

------------------------------------------------------------------------------
GET /events
------------------------------------------------------------------------------
Returns events with filtering options.

Parameters:
- apiKey (required): Your API key
- sport (required): Sport slug
- league (optional): League slug
- status (optional): Comma-separated statuses (pending, live, settled)
- from (optional): Start date/time in RFC3339 format
- to (optional): End date/time in RFC3339 format
- bookmaker (optional): Filter events that have odds from this bookmaker
- participantId (optional): Filter by team/participant ID

Example: /events?apiKey=YOUR_KEY&sport=football&status=pending,live

Response:
[
  {
    "id": 123456,
    "home": "Manchester United",
    "away": "Liverpool",
    "homeId": 38,
    "awayId": 42,
    "date": "2025-10-15T15:00:00Z",
    "status": "pending",
    "sport": {"name": "Football", "slug": "football"},
    "league": {"name": "England - Premier League", "slug": "england-premier-league"},
    "scores": {"home": 0, "away": 0}
  }
]

------------------------------------------------------------------------------
GET /events/live
------------------------------------------------------------------------------
Returns all currently live events.

Parameters:
- apiKey (required): Your API key
- sport (optional): Filter by sport slug

Example: /events/live?apiKey=YOUR_KEY&sport=football

------------------------------------------------------------------------------
GET /events/search
------------------------------------------------------------------------------
Search events by text query.

Parameters:
- apiKey (required): Your API key
- query (required): Search term (min 3 characters)

Example: /events/search?apiKey=YOUR_KEY&query=Liverpool

------------------------------------------------------------------------------
GET /events/{id}
------------------------------------------------------------------------------
Get a single event by ID.

Parameters:
- apiKey (required): Your API key
- id (required): Event ID in path

Example: /events/123456?apiKey=YOUR_KEY

------------------------------------------------------------------------------
GET /odds
------------------------------------------------------------------------------
Get odds for a specific event from selected bookmakers.

Parameters:
- apiKey (required): Your API key
- eventId (required): Event ID
- bookmakers (required): Comma-separated bookmaker names (max 30)

Example: /odds?apiKey=YOUR_KEY&eventId=123456&bookmakers=Bet365,SingBet,Unibet

Response:
{
  "id": 123456,
  "home": "Manchester United",
  "away": "Liverpool",
  "date": "2025-10-15T15:00:00Z",
  "status": "pending",
  "sport": {"name": "Football", "slug": "football"},
  "league": {"name": "England - Premier League", "slug": "england-premier-league"},
  "urls": {
    "Bet365": "https://bet365.com/...",
    "SingBet": "https://SingBet.com/..."
  },
  "bookmakers": {
    "Bet365": [
      {
        "name": "ML",
        "updatedAt": "2025-10-04T10:30:00Z",
        "odds": [
          {"home": "2.10", "draw": "3.40", "away": "3.20"}
        ]
      },
      {
        "name": "Asian Handicap",
        "updatedAt": "2025-10-04T10:30:00Z",
        "odds": [
          {"hdp": -0.5, "home": "1.95", "away": "1.85"},
          {"hdp": 0, "home": "1.75", "away": "2.05"}
        ]
      },
      {
        "name": "Totals",
        "updatedAt": "2025-10-04T10:30:00Z",
        "odds": [
          {"hdp": 2.5, "over": "1.90", "under": "1.90"}
        ]
      }
    ]
  }
}

Market Types:
- ML: Match result (home, draw, away)
- Asian Handicap: Handicap with hdp value
- Totals: Over/Under with hdp as the line
- Both Teams to Score: yes/no markets
- Correct Score: Exact score predictions
- And many more...

------------------------------------------------------------------------------
GET /odds/multi
------------------------------------------------------------------------------
Get odds for multiple events in a single request (counts as 1 API call).

Parameters:
- apiKey (required): Your API key
- eventIds (required): Comma-separated event IDs (max 10)
- bookmakers (required): Comma-separated bookmaker names (max 30)

Example: /odds/multi?apiKey=YOUR_KEY&eventIds=123456,123457,123458&bookmakers=Bet365,SingBet

Response: Array of event odds objects (same format as /odds)

------------------------------------------------------------------------------
GET /odds/updated
------------------------------------------------------------------------------
Get odds updated since a specific timestamp.

Parameters:
- apiKey (required): Your API key
- since (required): Unix timestamp (max 1 minute old)
- bookmaker (required): Bookmaker name
- sport (required): Sport name

Example: /odds/updated?apiKey=YOUR_KEY&since=1734450000&bookmaker=SingBet&sport=Football

------------------------------------------------------------------------------
GET /odds/movements
------------------------------------------------------------------------------
Get historical odds movements for a specific market.

Parameters:
- apiKey (required): Your API key
- eventId (required): Event ID
- bookmaker (required): Bookmaker name
- market (required): Market name (e.g., "ML", "Totals")
- marketLine (optional): Handicap line (required for non-ML markets)

Example: /odds/movements?apiKey=YOUR_KEY&eventId=123456&bookmaker=SingBet&market=ML

Response:
{
  "eventid": "123456",
  "bookmaker": "SingBet",
  "opening": {"home": 2.0, "draw": 3.2, "away": 3.5, "timestamp": 1734400000},
  "latest": {"home": 2.1, "draw": 3.3, "away": 3.4, "timestamp": 1734450000},
  "movements": [
    {"home": 2.0, "draw": 3.2, "away": 3.5, "timestamp": 1734400000},
    {"home": 2.05, "draw": 3.25, "away": 3.45, "timestamp": 1734420000},
    {"home": 2.1, "draw": 3.3, "away": 3.4, "timestamp": 1734450000}
  ]
}

------------------------------------------------------------------------------
GET /value-bets
------------------------------------------------------------------------------
Get value betting opportunities calculated by comparing bookmaker odds against market consensus.

Parameters:
- apiKey (required): Your API key
- bookmaker (required): Bookmaker name
- includeEventDetails (optional): Include event info (true/false)

Example: /value-bets?apiKey=YOUR_KEY&bookmaker=Bet365&includeEventDetails=true

Response:
[
  {
    "id": "vb_123456_bet365_home",
    "eventId": 123456,
    "bookmaker": "Bet365",
    "market": {
      "name": "ML",
      "home": "2.10",
      "draw": "3.40",
      "away": "3.20"
    },
    "betSide": "home",
    "expectedValue": 0.0523,
    "expectedValueUpdatedAt": "2025-10-04T10:30:00Z",
    "bookmakerOdds": {
      "home": "2.10",
      "draw": "3.40",
      "away": "3.20",
      "homeDirectLink": "https://bet365.com/..."
    },
    "event": {
      "home": "Manchester United",
      "away": "Liverpool",
      "date": "2025-10-15T15:00:00Z",
      "sport": "Football",
      "league": "England - Premier League"
    }
  }
]

Expected Value (EV) = (Probability × Odds) - 1
Positive EV indicates a value bet.

------------------------------------------------------------------------------
GET /arbitrage-bets
------------------------------------------------------------------------------
Get arbitrage betting opportunities across bookmakers.

Parameters:
- apiKey (required): Your API key
- bookmakers (required): Comma-separated bookmaker names
- limit (optional): Max results (default 50, max 500)
- includeEventDetails (optional): Include event info (true/false)

Example: /arbitrage-bets?apiKey=YOUR_KEY&bookmakers=Bet365,SingBet&includeEventDetails=true

Response:
[
  {
    "id": "arb_123456_ml",
    "eventId": 123456,
    "market": {"name": "ML", "hdp": null},
    "profitMargin": 0.0215,
    "impliedProbability": 0.9785,
    "totalStake": 100,
    "legs": [
      {"side": "home", "bookmaker": "Bet365", "odds": "2.15", "directLink": "..."},
      {"side": "away", "bookmaker": "SingBet", "odds": "2.10", "directLink": "..."}
    ],
    "optimalStakes": [
      {"side": "home", "bookmaker": "Bet365", "stake": 48.84, "potentialReturn": 105.00},
      {"side": "away", "bookmaker": "SingBet", "stake": 51.16, "potentialReturn": 107.44}
    ],
    "event": {...}
  }
]

------------------------------------------------------------------------------
GET /participants
------------------------------------------------------------------------------
Get teams/participants for a sport.

Parameters:
- apiKey (required): Your API key
- sport (required): Sport slug
- search (optional): Filter by name

Example: /participants?apiKey=YOUR_KEY&sport=football&search=Chelsea

Response:
[
  {"id": 45, "name": "Chelsea", "sport": "football"},
  {"id": 78, "name": "Chelsea Women", "sport": "football"}
]

------------------------------------------------------------------------------
GET /participants/{id}
------------------------------------------------------------------------------
Get participant by ID.

Example: /participants/45?apiKey=YOUR_KEY

------------------------------------------------------------------------------
GET /bookmakers/selected
------------------------------------------------------------------------------
Get bookmakers selected by the user.

Example: /bookmakers/selected?apiKey=YOUR_KEY

------------------------------------------------------------------------------
PUT /bookmakers/selected/select
------------------------------------------------------------------------------
Add bookmakers to user's selection.

Parameters:
- apiKey (required): Your API key
- bookmakers (required): Comma-separated bookmaker names

Example: /bookmakers/selected/select?apiKey=YOUR_KEY&bookmakers=Bet365,SingBet

------------------------------------------------------------------------------
PUT /bookmakers/selected/clear
------------------------------------------------------------------------------
Clear all selected bookmakers (limited to once every 12 hours).

Example: /bookmakers/selected/clear?apiKey=YOUR_KEY

================================================================================
WEBSOCKET API
================================================================================

Real-time odds updates via WebSocket connection.

Endpoint: wss://api.odds-api.io/v3/ws?apiKey=YOUR_API_KEY&markets=ML,Spread,Totals

Parameters:
- apiKey (required): Your API key
- markets (required): Comma-separated market names (max 20). Example: ML,Spread,Totals
- sport (optional): Comma-separated sport slugs (max 10). Example: football,basketball
- leagues (optional): Comma-separated league slugs (max 20). Example: england-premier-league
- eventIds (optional): Comma-separated event IDs (max 50). Example: 12345,67890
- status (optional): "live" or "prematch"

IMPORTANT:
- The markets parameter is required
- leagues and eventIds cannot be used together
- Using leagues or eventIds is recommended to reduce bandwidth

Example URLs:
wss://api.odds-api.io/v3/ws?apiKey=xxx&markets=ML,Spread,Totals&sport=football&status=live
wss://api.odds-api.io/v3/ws?apiKey=xxx&markets=ML&leagues=england-premier-league,spain-laliga
wss://api.odds-api.io/v3/ws?apiKey=xxx&markets=ML,Totals&eventIds=12345,67890

Message Types:
- welcome: Connection confirmed (includes active filters)
- created: New match added
- updated: Match or market changed
- deleted: Match removed
- no_markets: Match exists but no markets available

Welcome Message (sent on connection):
{
  "type": "welcome",
  "message": "Connected to OddsAPI WebSocket",
  "user_id": "user123",
  "bookmakers": ["Bet365", "Pinnacle"],
  "sport_filter": ["Football"],
  "leagues_filter": ["england-premier-league"],
  "status_filter": "live",
  "market_filter": ["ML", "SPREAD", "TOTALS"],
  "connected_at": "2026-01-15T21:00:00Z"
}

Update Message Format:
{
  "type": "updated",
  "timestamp": "2025-08-18T14:52:52.983Z",
  "id": "63017989",
  "bookie": "SingBet",
  "url": "https://www.singbet.com/sports/football/match/63017989",
  "markets": [
    {
      "name": "ML",
      "updatedAt": "2024-01-15T10:30:00Z",
      "odds": [{"home": "1.85", "draw": "3.25", "away": "2.10", "max": 500}]
    }
  ]
}

Error Responses (400 Bad Request):
- Too many sports. Maximum 10 allowed.
- Too many leagues. Maximum 20 allowed.
- Too many event IDs. Maximum 50 allowed.
- Too many markets. Maximum 20 allowed.
- Cannot use both 'leagues' and 'eventIds' filters together.
- Invalid status filter. Use 'prematch' or 'live'

Pricing: 2x the REST API price (add-on feature, subscribe at https://odds-api.io)

Bookmakers: The WebSocket automatically sends updates for all bookmakers you have selected
in your account. Manage bookmakers via PUT /bookmakers/selected/select endpoint.

================================================================================
SUPPORTED SPORTS (34 total)
================================================================================

Football (football)
Basketball (basketball)
Tennis (tennis)
Baseball (baseball)
American Football (american-football)
Ice Hockey (ice-hockey)
Esports (esports)
Darts (darts)
MMA (mixed-martial-arts)
Boxing (boxing)
Handball (handball)
Volleyball (volleyball)
Snooker (snooker)
Table Tennis (table-tennis)
Rugby (rugby)
Cricket (cricket)
Waterpolo (water-polo)
Futsal (futsal)
Beach Volley (beach-volleyball)
Aussie Rules (aussie-rules)
Floorball (floorball)
Squash (squash)
Beach Soccer (beach-soccer)
Lacrosse (lacrosse)
Curling (curling)
Padel (padel)
Bandy (bandy)
Gaelic Football (gaelic-football)
Beach Handball (beach-handball)
Athletics (athletics)
Badminton (badminton)
Cross-Country (cross-country)
Golf (golf)
Cycling (cycling)

================================================================================
CODE EXAMPLES
================================================================================

------------------------------------------------------------------------------
JavaScript/Node.js - Fetch Events and Odds
------------------------------------------------------------------------------

const apiKey = process.env.ODDS_API_KEY;

// Get football events
const eventsResponse = await fetch(
  `https://api.odds-api.io/v3/events?apiKey=${apiKey}&sport=football`
);
const events = await eventsResponse.json();

// Get odds for first event
const eventId = events[0].id;
const oddsResponse = await fetch(
  `https://api.odds-api.io/v3/odds?apiKey=${apiKey}&eventId=${eventId}&bookmakers=Bet365,SingBet`
);
const odds = await oddsResponse.json();

// Find best home odds
let bestHomeOdds = { bookmaker: null, odds: 0 };
for (const [bookmaker, markets] of Object.entries(odds.bookmakers)) {
  const ml = markets.find(m => m.name === 'ML');
  if (ml && parseFloat(ml.odds[0].home) > bestHomeOdds.odds) {
    bestHomeOdds = { bookmaker, odds: parseFloat(ml.odds[0].home) };
  }
}
console.log(`Best home odds: ${bestHomeOdds.odds} at ${bestHomeOdds.bookmaker}`);

------------------------------------------------------------------------------
Python - Fetch Value Bets
------------------------------------------------------------------------------

import requests
import os

api_key = os.environ['ODDS_API_KEY']

response = requests.get(
    'https://api.odds-api.io/v3/value-bets',
    params={
        'apiKey': api_key,
        'bookmaker': 'Bet365',
        'includeEventDetails': True
    }
)
value_bets = response.json()

# Filter bets with > 3% expected value
top_bets = [bet for bet in value_bets if bet['expectedValue'] >= 0.03]
top_bets.sort(key=lambda x: x['expectedValue'], reverse=True)

for bet in top_bets[:5]:
    print(f"{bet['event']['home']} vs {bet['event']['away']}")
    print(f"Bet on: {bet['betSide']}")
    print(f"EV: {bet['expectedValue'] * 100:.2f}%")
    print("---")

------------------------------------------------------------------------------
JavaScript - WebSocket Connection
------------------------------------------------------------------------------

const WebSocket = require('ws');

const apiKey = process.env.ODDS_API_KEY;
const params = new URLSearchParams({
  apiKey,
  markets: 'ML,Spread,Totals',
  sport: 'football',
  status: 'live'
});

const ws = new WebSocket(`wss://api.odds-api.io/v3/ws?${params}`);

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  switch (data.type) {
    case 'welcome':
      console.log(`Connected! Filters: ${data.sport_filter}, ${data.status_filter}`);
      break;
    case 'created':
      console.log(`New match: ${data.id}`);
      break;
    case 'updated':
      console.log(`Updated: ${data.id} - ${data.markets.length} markets`);
      break;
    case 'deleted':
      console.log(`Deleted: ${data.id}`);
      break;
  }
};

ws.onerror = (err) => console.error("Error:", err);
ws.onclose = () => console.log("Disconnected");

------------------------------------------------------------------------------
cURL Examples
------------------------------------------------------------------------------

# Get sports (no auth required)
curl "https://api.odds-api.io/v3/sports"

# Get bookmakers (no auth required)
curl "https://api.odds-api.io/v3/bookmakers"

# Get football events
curl "https://api.odds-api.io/v3/events?apiKey=YOUR_KEY&sport=football"

# Get odds for an event
curl "https://api.odds-api.io/v3/odds?apiKey=YOUR_KEY&eventId=123456&bookmakers=Bet365,SingBet"

# Get value bets
curl "https://api.odds-api.io/v3/value-bets?apiKey=YOUR_KEY&bookmaker=Bet365&includeEventDetails=true"

# Get arbitrage opportunities
curl "https://api.odds-api.io/v3/arbitrage-bets?apiKey=YOUR_KEY&bookmakers=Bet365,SingBet"

================================================================================
BEST PRACTICES
================================================================================

1. Use Multi-Odds Endpoint
   - Fetch up to 10 events in one request (counts as 1 API call)
   - Significantly reduces rate limit usage

2. Cache Responses Appropriately
   - Event lists: 5-10 minutes
   - Pre-match odds: 30-60 seconds
   - Live odds: 5-10 seconds
   - Use /odds/updated for incremental updates

3. Select Only Needed Bookmakers
   - Request only bookmakers your users need
   - Reduces response size and processing time

4. Filter Events by Bookmaker
   - Use ?bookmaker=Bet365 on /events endpoint to only get events with odds from that bookmaker
   - Much more efficient than fetching all events and filtering client-side
   - Example: /events?sport=football&bookmaker=Bet365&apiKey={key}

5. Handle Errors Gracefully
   - Implement exponential backoff for rate limits (429)
   - Not all bookmakers offer all markets

6. Monitor Rate Limits
   - Check response headers for remaining requests
   - Implement queuing for high-volume applications

================================================================================
CONTACT & SUPPORT
================================================================================

Email: hello@odds-api.io
Website: https://odds-api.io
Documentation: https://docs.odds-api.io
Bookmakers List: https://odds-api.io/sportsbooks

================================================================================
END OF DOCUMENTATION
================================================================================