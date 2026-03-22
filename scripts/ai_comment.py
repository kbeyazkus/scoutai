#!/usr/bin/env python3
"""
ai_comment.py
Reads footystats_today.json + footystats_tomorrow.json
Writes AI prematch comments for matches that don't have one yet.
Runs AFTER fetch.py — keeps fetch.py fast.
"""
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from google import genai
except Exception:
    genai = None

DATA_DIR             = 'data'
FOOTYSTATS_TODAY_JSON    = os.path.join(DATA_DIR, 'footystats_today.json')
FOOTYSTATS_TOMORROW_JSON = os.path.join(DATA_DIR, 'footystats_tomorrow.json')
SPORTMONKS_TODAY_JSON    = os.path.join(DATA_DIR, 'sportmonks_today.json')
SPORTMONKS_TOMORROW_JSON = os.path.join(DATA_DIR, 'sportmonks_tomorrow.json')
HEALTH_JSON              = os.path.join(DATA_DIR, 'health.json')

GEMINI_MODEL = (os.getenv('GEMINI_MODEL') or 'gemini-2.5-flash').strip()
# Max matches to write AI comments per run — prevents timeout
MAX_AI_PER_RUN = int(os.getenv('AI_MAX_PER_RUN', '40'))


def log(msg: str): print(msg, flush=True)

def safe_float(v: Any, default: float = 0.0) -> float:
    try: return float(v) if v not in (None, '') else default
    except Exception: return default

def load_json(path: str, default: Any):
    try:
        with open(path, 'r', encoding='utf-8') as f: return json.load(f)
    except Exception: return default

def save_json(path: str, data: Any):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def init_vertex_client():
    if genai is None: return None
    # Try credentials file first, fallback to env
    gac = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', '').strip()
    if gac and os.path.exists(gac):
        try:
            with open(gac, 'r', encoding='utf-8') as f:
                info = json.load(f)
            project = info.get('project_id')
            if project:
                return genai.Client(vertexai=True, project=project, location='us-central1')
        except Exception as e:
            log(f'Credentials read error: {e}')
    project  = os.getenv('GCP_PROJECT_ID', '').strip()
    location = os.getenv('GCP_LOCATION', 'us-central1').strip()
    if not project: return None
    try:
        return genai.Client(vertexai=True, project=project, location=location)
    except Exception as e:
        log(f'Gemini init error: {e}')
        return None

def ai_comment_prematch(client, match: Dict[str, Any]) -> str:
    if client is None: return ''
    payload = {
        'home':     match.get('home_name'),
        'away':     match.get('away_name'),
        'league':   match.get('competition_name'),
        'xg_home':  safe_float(match.get('team_a_xg_prematch')),
        'xg_away':  safe_float(match.get('team_b_xg_prematch')),
        'home_ppg': safe_float(match.get('home_ppg')),
        'away_ppg': safe_float(match.get('away_ppg')),
        'btts':     safe_float(match.get('btts_potential')),
        'over25':   safe_float(match.get('o25_potential')),
        'iy05':     safe_float(match.get('o05HT_potential')),
        'odds': {
            '1': safe_float(match.get('odds_ft_1')),
            'x': safe_float(match.get('odds_ft_x')),
            '2': safe_float(match.get('odds_ft_2')),
            'over25': safe_float(match.get('odds_ft_over25')),
            'btts':   safe_float(match.get('odds_btts_yes')),
        },
    }
    # Skip if no meaningful data
    if not payload['xg_home'] and not payload['home_ppg'] and not payload['btts']:
        return ''
    prompt = '\n'.join([
        'You are a technical betting analysis engine that cross-checks data.',
        'Cross-check xG, PPG, odds. If data conflicts flag as RISKY.',
        'Recommend only ONE lowest-risk bet from:',
        'Home Win, Away Win, Over 2.5, Both Teams Score,',
        'First Half Over 0.5, Asian Handicap,',
        'Home Team +1.5 Over Goals, Away Team +1.5 Over Goals.',
        'If confidence is low output exactly: "No Bet / Skip".',
        'Rules: Max 80 words. 3 sections: STATUS, REASON, CONCLUSION.',
        'No emojis, no flattery, sharp and technical.',
        json.dumps(payload, ensure_ascii=False),
    ])
    try:
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        text = (getattr(response, 'text', '') or '').strip()
        if len(text) > 500:
            text = text[:500].rsplit(' ', 1)[0]
        return text
    except Exception as e:
        log(f'AI error ({match.get("home_name")} - {match.get("away_name")}): {e}')
        return ''

def process_file(path: str, client, counter: dict) -> int:
    """Process one JSON file, write AI comments for matches missing them."""
    payload = load_json(path, {})
    rows: List[Dict] = payload.get('data', [])
    if not rows:
        log(f'⚠️  {path}: no data')
        return 0

    written = 0
    for row in rows:
        if counter['total'] >= MAX_AI_PER_RUN:
            break
        # Skip if already has a comment
        if row.get('boss_ai_decision') or row.get('prematch_comment'):
            continue
        # Skip matches with no statistical data
        if not safe_float(row.get('team_a_xg_prematch')) and \
           not safe_float(row.get('home_ppg')) and \
           not safe_float(row.get('btts_potential')):
            continue

        comment = ai_comment_prematch(client, row)
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
    client = init_vertex_client()
    if client is None:
        log('⚠️  Gemini client başlatılamadı — AI yorumlar atlanıyor')
        return

    health  = load_json(HEALTH_JSON, {})
    counter = {'total': 0}

    files = [
        FOOTYSTATS_TODAY_JSON,
        FOOTYSTATS_TOMORROW_JSON,
        SPORTMONKS_TODAY_JSON,
        SPORTMONKS_TOMORROW_JSON,
    ]

    total_written = 0
    for path in files:
        if counter['total'] >= MAX_AI_PER_RUN:
            log(f'ℹ️  MAX_AI_PER_RUN={MAX_AI_PER_RUN} limitine ulaşıldı, durduruluyor')
            break
        if not os.path.exists(path):
            log(f'⚠️  {path} bulunamadı, atlandı')
            continue
        total_written += process_file(path, client, counter)

    duration = round(time.time() - started, 2)
    health.update({
        'ai_runner':       'ai_comment',
        'ai_finished_at':  datetime.utcnow().isoformat() + 'Z',
        'ai_written':      total_written,
        'ai_duration_sec': duration,
    })
    save_json(HEALTH_JSON, health)

    log(f'✅ Toplam AI yorum: {total_written}')
    log(f'✅ Süre: {duration}s')

if __name__ == '__main__':
    main()
