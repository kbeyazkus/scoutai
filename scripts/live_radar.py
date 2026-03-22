#!/usr/bin/env python3
"""
live_radar.py
- Fetches live fixtures from Sportmonks
- Builds RICH AI payload (pressure, shots, predictions, standings, h2h, sidelined)
- Writes AI comment using full context
- Saves SLIM sportmonks_live.json (score + AI comment only) — prevents file bloat
- Saves full detail back to sportmonks_bundle.json for frontend
"""
import json, os, re, time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import requests
from unidecode import unidecode

try:
    from google import genai
except Exception:
    genai = None

DATA_DIR             = 'data'
FOOTYSTATS_TODAY_JSON = os.path.join(DATA_DIR, 'footystats_today.json')
FOOTYSTATS_LIVE_JSON  = os.path.join(DATA_DIR, 'footystats_live.json')
SPORTMONKS_LIVE_JSON  = os.path.join(DATA_DIR, 'sportmonks_live.json')
BUNDLE_JSON           = os.path.join(DATA_DIR, 'sportmonks_bundle.json')
HEALTH_JSON           = os.path.join(DATA_DIR, 'health.json')

SM_KEY         = os.getenv('SPORTMONKS_KEY','').strip()
GEMINI_MODEL   = (os.getenv('GEMINI_MODEL') or 'gemini-2.5-flash').strip()
REQUEST_TIMEOUT = 25
DETAIL_TTL_SEC  = 300

LIVE_INCLUDE = 'participants;scores;periods;events;league.country;round;state'
DETAIL_INCLUDE = (
    'participants;league.country;venue;state;scores;periods;'
    'events.type;events.period;events.player;'
    'statistics.type;lineups.player;lineups.type;'
    'coaches;sidelined.sideline.player;sidelined.sideline.type;'
    'weatherReport;predictions.type'
)

def log(msg): print(msg, flush=True)
def sf(v,d=0.0):
    try: return float(v) if v not in (None,'') else d
    except: return d
def si(v,d=0):
    try: return int(float(v)) if v not in (None,'') else d
    except: return d

def load_json(path, default):
    try:
        with open(path,'r',encoding='utf-8') as f: return json.load(f)
    except: return default

def save_json(path, data):
    with open(path,'w',encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def fetch_json(url):
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log(f'API error: {e}')
        return {}

def init_client():
    if genai is None: return None
    gac = os.getenv('GOOGLE_APPLICATION_CREDENTIALS','').strip()
    if gac and os.path.exists(gac):
        try:
            info = json.load(open(gac,'r',encoding='utf-8'))
            proj = info.get('project_id')
            if proj: return genai.Client(vertexai=True, project=proj, location='global')
        except Exception as e: log(f'Credentials error: {e}')
    proj = os.getenv('GCP_PROJECT_ID','').strip()
    loc  = os.getenv('GCP_LOCATION','global').strip()
    if not proj: return None
    try: return genai.Client(vertexai=True, project=proj, location=loc)
    except Exception as e: log(f'Vertex init error: {e}'); return None

def get_sides(parts):
    home = away = {}
    for p in parts:
        loc = str((p.get('meta') or {}).get('location') or '').lower()
        if loc == 'home': home = p
        elif loc == 'away': away = p
    if not home and parts: home = parts[0]
    if not away and len(parts) > 1: away = parts[1]
    return home, away

def current_score(fixture):
    for s in (fixture.get('scores') or []):
        if str(s.get('description','')).upper() == 'CURRENT':
            p = (s.get('score') or {}).get('participant')
            g = si((s.get('score') or {}).get('goals'))
            if p == 'home': h = g
            elif p == 'away': a = g
    try: return h, a
    except: pass
    for ev in (fixture.get('events') or []):
        result = ev.get('result')
        if result:
            m = re.match(r'^\s*(\d+)\s*-\s*(\d+)\s*$', str(result))
            if m: return int(m.group(1)), int(m.group(2))
    return 0, 0

def extract_minute(fixture):
    state = fixture.get('state') or {}
    m = si(state.get('minute'))
    if m: return m
    best = 0
    for p in (fixture.get('periods') or []):
        best = max(best, si(p.get('minutes')), si(p.get('minute')))
    for ev in (fixture.get('events') or []):
        best = max(best, si(ev.get('minute')))
    return best

def get_cached_detail(bundle, fid):
    cache = (bundle.get('fixtures') or {}).get(str(fid)) or {}
    ts = cache.get('fetched_at','')
    if ts:
        try:
            dt = datetime.fromisoformat(ts.replace('Z',''))
            if (datetime.utcnow() - dt).total_seconds() < DETAIL_TTL_SEC:
                return cache.get('detail') or {}
        except: pass
    return {}

def fetch_detail(fid):
    if not fid: return {}
    url = f'https://api.sportmonks.com/v3/football/fixtures/{fid}?api_token={SM_KEY}&include={DETAIL_INCLUDE}'
    return fetch_json(url).get('data') or {}

def extract_rich_context(detail, row):
    """Build rich AI context from full detail — xG, shots, pressure, predictions, sidelined."""
    # --- Shots ---
    shots_home = shots_away = shots_on_home = shots_on_away = 0
    for st in (detail.get('statistics') or []):
        dev = str((st.get('type') or {}).get('developer_name') or '').upper()
        loc = str(st.get('location') or '').lower()
        val = si((st.get('data') or {}).get('value'))
        if 'SHOTS_TOTAL' in dev:
            if loc == 'home': shots_home = val
            elif loc == 'away': shots_away = val
        elif 'SHOTS_ON_TARGET' in dev:
            if loc == 'home': shots_on_home = val
            elif loc == 'away': shots_on_away = val

    # --- Pressure (last 12 events) ---
    pressure_events = (detail.get('pressure') or [])[-12:]
    press_home = press_away = 0.0
    for p in pressure_events:
        loc = str((p.get('participant') or {}).get('meta',{}).get('location') or p.get('location') or '').lower()
        val = sf(p.get('value') or p.get('amount'))
        if loc == 'home': press_home += val
        else: press_away += val
    pressure_score = round(press_home - press_away, 2)

    # --- Predictions ---
    preds = {}
    for p in (detail.get('predictions') or []):
        dev = str((p.get('type') or {}).get('developer_name') or '').upper()
        vals = p.get('predictions') or {}
        if 'OVER_UNDER_2_5' in dev: preds['over25']   = sf(vals.get('yes'))
        elif 'BTTS' in dev:         preds['btts']     = sf(vals.get('yes'))
        elif 'HOME_WIN' in dev:     preds['home_win'] = sf(vals.get('yes'))
        elif 'AWAY_WIN' in dev:     preds['away_win'] = sf(vals.get('yes'))

    # --- Key sidelined ---
    sidelined = []
    for sl in (detail.get('sidelined') or [])[:5]:
        player = (sl.get('player') or (sl.get('sideline') or {}).get('player') or {})
        name = player.get('display_name') or player.get('name') or ''
        if name: sidelined.append(name)

    # --- Events summary ---
    goals = corners = yellows = reds = 0
    for ev in (detail.get('events') or []):
        t = str((ev.get('type') or {}).get('developer_name') or '').upper()
        if 'GOAL' in t: goals += 1
        elif 'CORNER' in t: corners += 1
        elif 'YELLOW' in t: yellows += 1
        elif 'RED' in t: reds += 1

    return {
        'shots_home': shots_home, 'shots_away': shots_away,
        'shots_on_home': shots_on_home, 'shots_on_away': shots_on_away,
        'pressure_score': pressure_score,
        'press_home': round(press_home, 1),
        'press_away': round(press_away, 1),
        'predictions': preds,
        'sidelined': sidelined,
        'events_summary': {
            'goals': goals, 'corners': corners,
            'yellows': yellows, 'reds': reds
        }
    }

def heuristic_comment(row, ctx):
    h = si(row.get('homeGoalCount'))
    a = si(row.get('awayGoalCount'))
    minute = si(row.get('elapsed'))
    press = sf(ctx.get('pressure_score'))
    preds = ctx.get('predictions') or {}
    sh = ctx.get('shots_on_home',0)
    sa = ctx.get('shots_on_away',0)

    if h == a:
        if press >= 8 or sh >= sa + 2:
            return f"STATUS: Score level {h}-{a} at {minute}'.\\nREASON: Home pressure ({ctx.get('press_home',0)}) dominates, {sh} shots on target.\\nCONCLUSION: Home goal or Over 2.5 worth monitoring."
        if press <= -8 or sa >= sh + 2:
            return f"STATUS: Score level {h}-{a} at {minute}'.\\nREASON: Away pressure ({ctx.get('press_away',0)}) dominates, {sa} shots on target.\\nCONCLUSION: Away goal or Over 2.5 worth monitoring."
        if preds.get('over25',0) >= 60 or preds.get('btts',0) >= 60:
            return f"STATUS: Score level {h}-{a}.\\nREASON: Model maintains high-scoring tendency (O2.5: {preds.get('over25',0):.0f}%, BTTS: {preds.get('btts',0):.0f}%).\\nCONCLUSION: Over 2.5 or Both Teams Score worth monitoring."
        return f"STATUS: Evenly contested, score {h}-{a} at {minute}'.\\nREASON: No clear pressure or shot dominance.\\nCONCLUSION: No Bet / Skip for now."
    leader  = row.get('home_name') if h > a else row.get('away_name')
    trailer = row.get('away_name') if h > a else row.get('home_name')
    if abs(press) >= 8:
        return f"STATUS: {leader} leads {h}-{a} but pressure shifting.\\nREASON: {trailer} pressure building, match still open.\\nCONCLUSION: Live monitoring recommended."
    return f"STATUS: {leader} leads {h}-{a} at {minute}'.\\nREASON: {trailer} produced no clear response.\\nCONCLUSION: Current direction maintained."

def ai_comment_live(client, row, detail, ctx):
    heuristic = heuristic_comment(row, ctx)
    if client is None: return heuristic
    minute = si(row.get('elapsed'))
    if minute < 65: return heuristic

    payload = {
        'home':           row.get('home_name'),
        'away':           row.get('away_name'),
        'minute':         minute,
        'score':          f"{si(row.get('homeGoalCount'))}-{si(row.get('awayGoalCount'))}",
        'pressure_score': ctx.get('pressure_score'),
        'pressure_home':  ctx.get('press_home'),
        'pressure_away':  ctx.get('press_away'),
        'shots_home':     ctx.get('shots_home'),
        'shots_away':     ctx.get('shots_away'),
        'shots_on_target_home': ctx.get('shots_on_home'),
        'shots_on_target_away': ctx.get('shots_on_away'),
        'events_summary': ctx.get('events_summary'),
        'model_predictions': ctx.get('predictions'),
        'key_sidelined':  ctx.get('sidelined'),
        'home_ppg':       sf(row.get('home_ppg')),
        'away_ppg':       sf(row.get('away_ppg')),
        'btts_potential': sf(row.get('btts_potential')),
        'o25_potential':  sf(row.get('o25_potential')),
    }

    prompt = '\n'.join([
        'You are a technical live betting analysis engine.',
        'Cross-check ALL provided metrics: pressure, shots on target, model predictions, PPG.',
        'If data conflicts flag as RISKY.',
        'Recommend ONE lowest-risk bet from:',
        'Home Win, Away Win, Over 2.5, Both Teams Score,',
        'First Half Over 0.5, Asian Handicap,',
        'Home Team +1.5 Over Goals, Away Team +1.5 Over Goals.',
        'If confidence is low output exactly: "No Bet / Skip".',
        'Rules: Max 80 words. 3 sections: STATUS, REASON, CONCLUSION.',
        'Live match 65+ min: weight live pressure and shots over prematch stats.',
        'No emojis, no flattery, sharp and technical.',
        '',
        json.dumps(payload, ensure_ascii=False),
    ])

    for attempt in range(3):
        try:
            resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            text = (getattr(resp,'text','') or '').strip()
            return text[:520].rsplit(' ',1)[0] if len(text) > 520 else (text or heuristic)
        except Exception as e:
            err = str(e)
            if '429' in err or 'RESOURCE_EXHAUSTED' in err:
                wait = 10 * (2 ** attempt)
                log(f'⚠️  429 — waiting {wait}s (attempt {attempt+1}/3)')
                time.sleep(wait)
            else:
                log(f'Live AI error ({row.get("home_name")} - {row.get("away_name")}): {e}')
                return heuristic
    return heuristic

def build_slim_row(fixture, detail, row_base, ctx, live_comment):
    """Build SLIM output row — only what frontend needs for live display.
    Full detail stays in bundle.json — this file stays small."""
    h, a = current_score(fixture)
    minute = extract_minute(fixture)
    parts = fixture.get('participants') or []
    home, away = get_sides(parts)
    league = fixture.get('league') or {}

    return {
        'id':              f"sm-{fixture.get('id')}",
        'source_ids':      {'footystats': '', 'sportmonks': str(fixture.get('id') or '')},
        'home_name':       home.get('name','Home'),
        'away_name':       away.get('name','Away'),
        'home_image':      home.get('image_path',''),
        'away_image':      away.get('image_path',''),
        'competition_name': league.get('name',''),
        'league_country':  (league.get('country') or {}).get('name',''),
        'league_country_image': (league.get('country') or {}).get('image_path',''),
        'home_country':    (league.get('country') or {}).get('name',''),
        'away_country':    (league.get('country') or {}).get('name',''),
        'date_unix':       si(fixture.get('starting_at_timestamp')),
        'homeGoalCount':   h,
        'awayGoalCount':   a,
        'status':          'live',
        'elapsed':         minute,
        'pressure_score':  ctx.get('pressure_score', 0),
        'shots_home':      ctx.get('shots_home', 0),
        'shots_away':      ctx.get('shots_away', 0),
        'live_comment':    live_comment,
        'boss_ai_decision': live_comment,
        'ai_comment':      live_comment,
        # Prematch data from base row if available
        'home_ppg':        sf(row_base.get('home_ppg')),
        'away_ppg':        sf(row_base.get('away_ppg')),
        'btts_potential':  sf(row_base.get('btts_potential')),
        'o25_potential':   sf(row_base.get('o25_potential')),
        'o05HT_potential': sf(row_base.get('o05HT_potential')),
        'odds_ft_1':       sf(row_base.get('odds_ft_1')),
        'odds_ft_x':       sf(row_base.get('odds_ft_x')),
        'odds_ft_2':       sf(row_base.get('odds_ft_2')),
    }

def find_prematch_row(today_rows, fixture):
    """Match live fixture to today's prematch row by name."""
    parts = fixture.get('participants') or []
    home, away = get_sides(parts)
    hname = (home.get('name') or '').lower()
    aname = (away.get('name') or '').lower()
    for r in today_rows:
        rh = (r.get('home_name') or '').lower()
        ra = (r.get('away_name') or '').lower()
        if hname[:5] in rh and aname[:5] in ra:
            return r
    return {}

def main():
    if not SM_KEY: return
    client  = init_client()
    bundle  = load_json(BUNDLE_JSON, {'fixtures': {}})
    health  = load_json(HEALTH_JSON, {})
    today_rows = load_json(FOOTYSTATS_TODAY_JSON, {}).get('data', [])

    url = (f'https://api.sportmonks.com/v3/football/livescores/inplay'
           f'?api_token={SM_KEY}&include={LIVE_INCLUDE}')
    live_fixtures = fetch_json(url).get('data') or []

    health.update({
        'live_runner':       'live_radar_rich',
        'live_started_at':   datetime.utcnow().isoformat() + 'Z',
        'live_fixtures_seen': len(live_fixtures),
        'live_ai_written':   0,
    })

    sm_live_out = []

    for fixture in live_fixtures:
        fid = str(fixture.get('id') or '')

        detail = get_cached_detail(bundle, fid)
        if not detail:
            detail = fetch_detail(si(fid))
            if detail:
                bundle.setdefault('fixtures', {})[fid] = {
                    'detail':     detail,
                    'fetched_at': datetime.utcnow().isoformat() + 'Z',
                }

        ctx = extract_rich_context(detail, {}) if detail else {
            'pressure_score': 0, 'press_home': 0, 'press_away': 0,
            'shots_home': 0, 'shots_away': 0, 'shots_on_home': 0, 'shots_on_away': 0,
            'predictions': {}, 'sidelined': [], 'events_summary': {}
        }

        # Build temp row for heuristic/AI
        h, a   = current_score(fixture)
        minute = extract_minute(fixture)
        parts  = fixture.get('participants') or []
        home_p, away_p = get_sides(parts)
        row_base = find_prematch_row(today_rows, fixture)
        temp_row = {
            **row_base,
            'home_name':      home_p.get('name','Home'),
            'away_name':      away_p.get('name','Away'),
            'homeGoalCount':  h,
            'awayGoalCount':  a,
            'elapsed':        minute,
            'status':         'live',
        }

        live_comment = ai_comment_live(client, temp_row, detail or {}, ctx)
        if live_comment:
            health['live_ai_written'] += 1

        slim = build_slim_row(fixture, detail or {}, row_base, ctx, live_comment)
        sm_live_out.append(slim)
        log(f'  📡 {slim["home_name"]} {h}-{a} {slim["away_name"]} | {minute}\' | AI: {"✅" if live_comment else "⏭"}')

    # FootyStats live
    fs_live = [x for x in today_rows
               if str(x.get('status','')).lower() in ('live','inplay','ht')
               or si(x.get('elapsed')) > 0]

    save_json(FOOTYSTATS_LIVE_JSON, {
        'data': fs_live,
        'updated_at': datetime.utcnow().isoformat() + 'Z'
    })
    save_json(SPORTMONKS_LIVE_JSON, {
        'matches': sm_live_out,
        'updated_at': datetime.utcnow().isoformat() + 'Z'
    })
    save_json(BUNDLE_JSON, bundle)
    health['live_finished_at'] = datetime.utcnow().isoformat() + 'Z'
    save_json(HEALTH_JSON, health)

    log(f'✅ Canlı maç: {len(sm_live_out)} | AI yorum: {health["live_ai_written"]}')

if __name__ == '__main__':
    main()
