#!/usr/bin/env python3
"""
ai_comment.py
Reads footystats + sportmonks JSON files + sportmonks_bundle.json
Builds rich AI payload per match and writes prematch AI comments.
Runs AFTER fetch.py — keeps fetch.py fast.
"""
import json, os, time
from datetime import datetime
from typing import Any, Dict, List

try:
    from google import genai
except Exception:
    genai = None

DATA_DIR              = 'data'
FS_TODAY_JSON         = os.path.join(DATA_DIR, 'footystats_today.json')
FS_TOMORROW_JSON      = os.path.join(DATA_DIR, 'footystats_tomorrow.json')
SM_TODAY_JSON         = os.path.join(DATA_DIR, 'sportmonks_today.json')
SM_TOMORROW_JSON      = os.path.join(DATA_DIR, 'sportmonks_tomorrow.json')
BUNDLE_JSON           = os.path.join(DATA_DIR, 'sportmonks_bundle.json')
HEALTH_JSON           = os.path.join(DATA_DIR, 'health.json')

GEMINI_MODEL  = (os.getenv('GEMINI_MODEL') or 'gemini-2.5-flash').strip()
MAX_AI_PER_RUN = int(os.getenv('AI_MAX_PER_RUN', '40'))

def log(msg: str): print(msg, flush=True)
def sf(v, d=0.0):
    try: return float(v) if v not in (None,'') else d
    except: return d
def si(v, d=0):
    try: return int(float(v)) if v not in (None,'') else d
    except: return d

def load_json(path, default):
    try:
        with open(path,'r',encoding='utf-8') as f: return json.load(f)
    except: return default

def save_json(path, data):
    with open(path,'w',encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

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
    except Exception as e: log(f'Gemini init error: {e}'); return None

def extract_bundle_data(bundle: Dict, row: Dict) -> Dict:
    """Extract rich context from sportmonks_bundle for AI payload."""
    fixtures = bundle.get('fixtures') or {}
    smid = str((row.get('source_ids') or {}).get('sportmonks') or row.get('sportmonks_id') or '')
    bdata = fixtures.get(smid) or {}

    # Fallback: name-based match
    if not bdata:
        home_key = (row.get('home_name') or '').lower()[:6]
        away_key = (row.get('away_name') or '').lower()[:6]
        for fdata in fixtures.values():
            d = (fdata.get('detail') or {})
            parts = [p.get('name','').lower() for p in (d.get('participants') or [])]
            if any(home_key in p for p in parts) and any(away_key in p for p in parts):
                bdata = fdata
                break

    detail     = bdata.get('detail') or {}
    standings  = bdata.get('standings') or []
    h2h        = bdata.get('h2h') or []
    odds_list  = bdata.get('odds') or []

    # --- Standings: home & away position ---
    home_pos = away_pos = home_pts = away_pts = None
    for s in standings:
        pname = (s.get('participant') or {}).get('name','').lower()
        home_n = (row.get('home_name') or '').lower()
        away_n = (row.get('away_name') or '').lower()
        if home_n[:5] in pname:
            home_pos = s.get('position'); home_pts = s.get('points')
        if away_n[:5] in pname:
            away_pos = s.get('position'); away_pts = s.get('points')

    # --- H2H: last 3 results ---
    h2h_summary = []
    for m in h2h[:3]:
        scores = m.get('scores') or []
        h = a = '-'
        for sc in scores:
            if str(sc.get('description','')).upper() == 'CURRENT':
                p = (sc.get('score') or {}).get('participant')
                g = (sc.get('score') or {}).get('goals')
                if p == 'home': h = g
                elif p == 'away': a = g
        h2h_summary.append(f"{h}-{a}")

    # --- Key sidelined players ---
    sidelined_names = []
    for sl in (detail.get('sidelined') or [])[:6]:
        player = (sl.get('player') or (sl.get('sideline') or {}).get('player') or {})
        name = player.get('display_name') or player.get('name') or ''
        if name: sidelined_names.append(name)

    # --- Predictions from bundle ---
    preds = {}
    for p in (detail.get('predictions') or []):
        dev = str((p.get('type') or {}).get('developer_name') or '').upper()
        vals = p.get('predictions') or {}
        if 'OVER_UNDER_2_5' in dev: preds['over25'] = sf(vals.get('yes'))
        elif 'BTTS' in dev:         preds['btts']   = sf(vals.get('yes'))
        elif 'HOME_WIN' in dev:     preds['home_win']= sf(vals.get('yes'))
        elif 'AWAY_WIN' in dev:     preds['away_win']= sf(vals.get('yes'))

    # --- Best odds from bundle ---
    best_odds = {}
    for o in odds_list[:20]:
        label = str(o.get('label') or o.get('name') or '').upper()
        val   = sf(o.get('value') or o.get('odds'))
        if '1' == label and not best_odds.get('1'):   best_odds['1'] = val
        elif 'X' == label and not best_odds.get('x'): best_odds['x'] = val
        elif '2' == label and not best_odds.get('2'): best_odds['2'] = val
        elif 'OVER' in label and '2.5' in label:      best_odds['over25'] = val
        elif 'BTTS' in label or 'BOTH' in label:      best_odds['btts'] = val

    # --- Weather ---
    weather = detail.get('weatherreport') or detail.get('weatherReport') or {}
    weather_str = weather.get('description') or ''

    return {
        'standings':  {'home_pos': home_pos, 'home_pts': home_pts,
                       'away_pos': away_pos, 'away_pts': away_pts},
        'h2h_last3':  h2h_summary,
        'sidelined':  sidelined_names,
        'predictions': preds,
        'bundle_odds': best_odds,
        'weather':    weather_str,
        'venue':      (detail.get('venue') or {}).get('name',''),
    }

def build_ai_payload(match: Dict, bundle_ctx: Dict) -> Dict:
    """Build complete AI payload combining footystats + bundle data."""
    return {
        'home':          match.get('home_name'),
        'away':          match.get('away_name'),
        'league':        match.get('competition_name'),
        'venue':         bundle_ctx.get('venue') or match.get('stadium_name',''),
        'weather':       bundle_ctx.get('weather',''),
        'xg_home':       sf(match.get('team_a_xg_prematch')),
        'xg_away':       sf(match.get('team_b_xg_prematch')),
        'home_ppg':      sf(match.get('home_ppg') or match.get('pre_match_home_ppg')),
        'away_ppg':      sf(match.get('away_ppg') or match.get('pre_match_away_ppg')),
        'btts_potential':sf(match.get('btts_potential')),
        'o25_potential': sf(match.get('o25_potential')),
        'iy05_potential':sf(match.get('o05HT_potential')),
        'odds': {
            '1':      sf(match.get('odds_ft_1'))      or bundle_ctx['bundle_odds'].get('1',0),
            'x':      sf(match.get('odds_ft_x'))      or bundle_ctx['bundle_odds'].get('x',0),
            '2':      sf(match.get('odds_ft_2'))      or bundle_ctx['bundle_odds'].get('2',0),
            'over25': sf(match.get('odds_ft_over25')) or bundle_ctx['bundle_odds'].get('over25',0),
            'btts':   sf(match.get('odds_btts_yes'))  or bundle_ctx['bundle_odds'].get('btts',0),
        },
        'model_predictions': bundle_ctx.get('predictions',{}),
        'standings': bundle_ctx.get('standings',{}),
        'h2h_last3': bundle_ctx.get('h2h_last3',[]),
        'key_sidelined': bundle_ctx.get('sidelined',[]),
    }

def ai_comment_prematch(client, match: Dict, bundle: Dict) -> str:
    if client is None: return ''

    ctx     = extract_bundle_data(bundle, match)
    payload = build_ai_payload(match, ctx)

    # Skip if zero data
    if (not payload['xg_home'] and not payload['home_ppg'] and
        not payload['btts_potential'] and not payload['model_predictions']):
        return ''

    prompt = '\n'.join([
        'You are a technical betting analysis engine. Cross-check ALL provided data.',
        'Data includes: xG, PPG, prematch potentials, bookmaker odds, model predictions,',
        'league standings, last 3 H2H results, and key sidelined players.',
        'Cross-check everything. If data sources conflict, flag as RISKY.',
        'Recommend ONE lowest-risk bet from:',
        'Home Win, Away Win, Over 2.5, Both Teams Score,',
        'First Half Over 0.5, Asian Handicap,',
        'Home Team +1.5 Over Goals, Away Team +1.5 Over Goals.',
        'If confidence is low output exactly: "No Bet / Skip".',
        'Rules: Max 80 words. 3 sections only: STATUS, REASON, CONCLUSION.',
        'No emojis, no flattery, sharp and technical.',
        '',
        json.dumps(payload, ensure_ascii=False),
    ])

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            text = (getattr(response,'text','') or '').strip()
            return text[:500].rsplit(' ',1)[0] if len(text) > 500 else text
        except Exception as e:
            err = str(e)
            if '429' in err or 'RESOURCE_EXHAUSTED' in err:
                wait = 15 * (2 ** attempt)
                log(f'⚠️  429 — waiting {wait}s (attempt {attempt+1}/{max_retries})')
                time.sleep(wait)
            else:
                log(f'AI error ({match.get("home_name")} - {match.get("away_name")}): {e}')
                return ''
    log('⚠️  AI skipped after retries')
    return ''

def process_file(path: str, client, bundle: Dict, counter: dict) -> int:
    payload = load_json(path, {})
    rows: List[Dict] = payload.get('data', [])
    if not rows:
        log(f'⚠️  {path}: no data')
        return 0

    written = 0
    for row in rows:
        if counter['total'] >= MAX_AI_PER_RUN: break
        if row.get('boss_ai_decision') or row.get('prematch_comment'): continue
        if (not sf(row.get('team_a_xg_prematch')) and
            not sf(row.get('home_ppg')) and
            not sf(row.get('btts_potential'))):
            continue

        comment = ai_comment_prematch(client, row, bundle)
        if comment:
            row['boss_ai_decision'] = comment
            row['prematch_comment'] = comment
            row['ai_comment']       = comment
            written += 1
            counter['total'] += 1
            log(f'  ✍️  {row.get("home_name")} - {row.get("away_name")}')

    if written:
        payload['updated_at'] = datetime.utcnow().isoformat() + 'Z'
        save_json(path, payload)
        log(f'✅ {path}: {written} AI yorum yazıldı')
    else:
        log(f'ℹ️  {path}: yeni yorum gerekmedi')
    return written

def main():
    started = time.time()
    client = init_client()
    if client is None:
        log('⚠️  Gemini client başlatılamadı — AI yorumlar atlanıyor')
        return

    bundle  = load_json(BUNDLE_JSON, {'fixtures': {}})
    health  = load_json(HEALTH_JSON, {})
    counter = {'total': 0}

    files = [FS_TODAY_JSON, FS_TOMORROW_JSON, SM_TODAY_JSON, SM_TOMORROW_JSON]

    total_written = 0
    for path in files:
        if counter['total'] >= MAX_AI_PER_RUN:
            log(f'ℹ️  MAX_AI_PER_RUN={MAX_AI_PER_RUN} limitine ulaşıldı')
            break
        if not os.path.exists(path):
            log(f'⚠️  {path} bulunamadı, atlandı')
            continue
        total_written += process_file(path, client, bundle, counter)

    duration = round(time.time() - started, 2)
    health.update({
        'ai_runner':       'ai_comment_rich',
        'ai_finished_at':  datetime.utcnow().isoformat() + 'Z',
        'ai_written':      total_written,
        'ai_duration_sec': duration,
    })
    save_json(HEALTH_JSON, health)
    log(f'✅ Toplam AI yorum: {total_written}')
    log(f'✅ Süre: {duration}s')

if __name__ == '__main__':
    main()
