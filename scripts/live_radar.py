import json, os, re, time, difflib
from pathlib import Path
from datetime import datetime, timezone
import requests
from unidecode import unidecode
from google import genai

SM_KEY = os.environ.get("SPORTMONKS_KEY", "").strip()
if not SM_KEY:
    raise SystemExit("SPORTMONKS_KEY secret eksik.")

creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
if not creds_path or not os.path.exists(creds_path):
    raise SystemExit("GOOGLE_APPLICATION_CREDENTIALS bulunamadi.")

with open(creds_path, 'r', encoding='utf-8') as f:
    info = json.load(f)
client = genai.Client(vertexai=True, project=info['project_id'], location="us-central1")

today_path = Path('data/today.json')
live_path = Path('data/live.json')

if not today_path.exists():
    raise SystemExit("data/today.json yok. Once fetch workflow calistirilmali.")

fs_data = json.loads(today_path.read_text(encoding='utf-8')).get('data', [])
if not isinstance(fs_data, list):
    fs_data = []

LIVE_STATES = {'INPLAY_1ST_HALF', 'INPLAY_2ND_HALF', 'HT', 'INPLAY_ET', 'INPLAY_ET_2', 'BREAK'}

url = (
    "https://api.sportmonks.com/v3/football/livescores"
    "?api_token=" + SM_KEY +
    "&include=participants;scores;state;statistics.type;periods"
)
res = requests.get(url, timeout=30)
res.raise_for_status()
live_matches = res.json().get('data', [])
live_matches = [m for m in live_matches if m.get('state', {}).get('developer_name', '') in LIVE_STATES]
print("Canli mac sayisi: " + str(len(live_matches)))

if not live_matches:
    live_path.write_text(
        json.dumps({"matches": [], "resultSet": {"count": 0}}, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )
    print("Su an canli mac yok.")
    raise SystemExit(0)


def clean_name(name):
    name = unidecode(str(name or '')).lower()
    name = re.sub(r'[^a-z0-9 ]+', ' ', name)
    words = [w for w in name.split() if w not in {'fc', 'cf', 'club', 'de', 'la', 'real', 'sk', 'fk', 'ac', 'sc', 'the'}]
    return ''.join(words)


def get_participant(parts, location):
    for p in parts:
        meta = p.get('meta', {}) or {}
        if meta.get('location') == location:
            return p
    if location == 'home':
        return parts[0] if parts else {}
    return parts[1] if len(parts) > 1 else {}


def get_score(scores, side):
    for s in scores:
        obj = s.get('score', {}) or {}
        if obj.get('participant') == side and s.get('description') == 'CURRENT':
            return int(obj.get('goals') or 0)
    return 0


def estimate_minute(match, state_name):
    if state_name in {'HT', 'BREAK'}:
        return 'HY'
    for v in [
        match.get('minute'),
        (match.get('time') or {}).get('minute'),
        (match.get('currentPeriod') or {}).get('minutes'),
    ]:
        if v is None:
            continue
        try:
            v = int(v)
            if v > 0:
                return v
        except Exception:
            pass
    ts = match.get('starting_at_timestamp')
    if ts:
        try:
            total = max(1, int((time.time() - int(ts)) / 60))
            if state_name == 'INPLAY_1ST_HALF':
                return min(total, 45)
            if state_name in {'INPLAY_2ND_HALF', 'INPLAY_ET', 'INPLAY_ET_2'}:
                return min(max(46, total - 15), 120)
        except Exception:
            pass
    return 0


def find_fs_match(sm_home, sm_away, fs_list):
    sm_h = clean_name(sm_home)
    sm_a = clean_name(sm_away)
    best, best_score = None, 0.0
    for item in fs_list:
        fh = clean_name(item.get('home_name', ''))
        fa = clean_name(item.get('away_name', ''))
        hr = difflib.SequenceMatcher(None, sm_h, fh).ratio()
        ar = difflib.SequenceMatcher(None, sm_a, fa).ratio()
        combo = (hr + ar) / 2
        if sm_h == fh and sm_a == fa:
            return item, 1.0
        if ((sm_h in fh or fh in sm_h) and (sm_a in fa or fa in sm_a)) and combo >= 0.55:
            if combo > best_score:
                best, best_score = item, combo
        elif hr >= 0.72 and ar >= 0.72 and combo > best_score:
            best, best_score = item, combo
    return best, best_score


ALL_DECISIONS = [
    'MS1', 'MSX', 'MS2', '1X', 'X2', '12',
    'Ust 0.5', 'Ust 1.5', 'Ust 2.5', 'Ust 3.5', 'Alt 2.5', 'Alt 1.5',
    'KG Var', 'KG Yok',
    'Ev -0.5', 'Ev -1', 'Ev -1.5', 'Ev +0.5', 'Ev +1', 'Ev +1.5',
    'Depl -0.5', 'Depl -1', 'Depl -1.5', 'Depl +0.5', 'Depl +1', 'Depl +1.5',
    'IY MS1', 'IY MSX', 'IY MS2', 'IY Ust 0.5', 'IY Ust 1.5',
    'Siradaki gol ev', 'Siradaki gol deplasman', 'Sonraki korner', 'Kalan sure gol cikar',
    'PAS GEC'
]

schema = {
    "type": "object",
    "properties": {
        "status_summary": {"type": "string"},
        "final_decision": {"type": "string", "enum": ALL_DECISIONS},
        "bankroll_percent": {"type": "integer", "minimum": 0, "maximum": 10},
        "risk_score": {"type": "integer", "minimum": 1, "maximum": 10},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        "reason_codes": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "PREMATCH_EDGE", "LIVE_PRESSURE", "LOW_DATA", "CONFLICT",
                    "SCORE_STATE", "ODDS_SIGNAL", "SECOND_HALF", "FIRST_HALF", "NO_VALUE"
                ]
            }
        }
    },
    "required": ["status_summary", "final_decision", "bankroll_percent", "risk_score", "confidence", "reason_codes"]
}


def build_allowed(state_name, h_score, a_score, has_stats):
    allowed = set(ALL_DECISIONS)
    if state_name not in {'INPLAY_1ST_HALF'}:
        allowed -= {'IY MS1', 'IY MSX', 'IY MS2', 'IY Ust 0.5', 'IY Ust 1.5'}
    if not has_stats:
        allowed -= {'Siradaki gol ev', 'Siradaki gol deplasman', 'Sonraki korner', 'Kalan sure gol cikar'}
    if h_score + a_score >= 4:
        allowed -= {'Alt 2.5', 'Alt 1.5'}
    return sorted(allowed)


live_payload = {"matches": [], "resultSet": {"count": len(live_matches)}}
log = {"total": len(live_matches), "matched": 0, "scored": 0, "ai_ok": 0, "pas_gec": 0, "errors": 0}

for lm in live_matches:
    parts = lm.get('participants', []) or []
    if len(parts) < 2:
        continue

    home_p = get_participant(parts, 'home')
    away_p = get_participant(parts, 'away')
    sm_home = home_p.get('name', '')
    sm_away = away_p.get('name', '')
    state_name = lm.get('state', {}).get('developer_name', '')
    scores = lm.get('scores', []) or []
    h_score = get_score(scores, 'home')
    a_score = get_score(scores, 'away')
    minute = estimate_minute(lm, state_name)

    live_payload['matches'].append({
        "homeTeam": {"name": sm_home},
        "awayTeam": {"name": sm_away},
        "score": {"fullTime": {"home": h_score, "away": a_score}},
        "minute": None if minute == 'HY' else minute,
        "state": state_name,
    })

    matched, ratio = find_fs_match(sm_home, sm_away, fs_data)
    if not matched:
        print("Eslesmedi: " + sm_home + " vs " + sm_away)
        continue

    log['matched'] += 1
    matched['homeGoalCount'] = h_score
    matched['awayGoalCount'] = a_score
    matched['totalGoalCount'] = h_score + a_score
    matched['status'] = 'live'
    matched['elapsed'] = minute
    matched['live_last_update'] = datetime.now(timezone.utc).isoformat()
    log['scored'] += 1

    stats = lm.get('statistics', []) or []
    stat_names = []
    for s in stats[:8]:
        typ = s.get('type') or {}
        n = typ.get('developer_name') or typ.get('name')
        if n:
            stat_names.append(str(n))
    has_stats = len(stats) > 0
    matched['live_stats_summary'] = (
        "Istatistik: " + str(len(stats)) + " | " + ', '.join(stat_names[:5])
        if has_stats else "Canli istatistik yok"
    )

    xg_h = matched.get('team_a_xg_prematch', 0)
    xg_a = matched.get('team_b_xg_prematch', 0)
    odds_1 = matched.get('odds_ft_1', '-')
    odds_x = matched.get('odds_ft_x', '-')
    odds_2 = matched.get('odds_ft_2', '-')
    ppg_h = matched.get('home_ppg', matched.get('team_a_ppg', 0))
    ppg_a = matched.get('away_ppg', matched.get('team_b_ppg', 0))
    btts = matched.get('btts_potential', 0)
    over25 = matched.get('o25_potential', matched.get('over25_potential', 0))
    allowed = build_allowed(state_name, h_score, a_score, has_stats)
    live_weight = 90 if state_name in {'INPLAY_2ND_HALF', 'INPLAY_ET', 'INPLAY_ET_2'} else 55
    prematch_weight = 10 if live_weight == 90 else 45

    prompt = "\n".join([
        "Sen disiplinli bir canli bahis karar motorusun.",
        "Kurallar:",
        "- Sadece ALLOWED_DECISIONS listesinden BIR secim yap veya PAS GEC sec.",
        "- Enum disinda hicbir deger uretme.",
        "- Veriler celiskiliyse PAS GEC.",
        "- Ikinci yarida live_weight yuksektir, canli sinyalleri daha fazla oemse.",
        "- Canli istatistik yoksa canli aksiyon marketlerini zorlama.",
        "",
        "MAC: " + sm_home + " vs " + sm_away,
        "PREMATCH:",
        "- xG ev/dep: " + str(xg_h) + " / " + str(xg_a),
        "- odds 1/X/2: " + str(odds_1) + " / " + str(odds_x) + " / " + str(odds_2),
        "- home_ppg: " + str(ppg_h) + " | away_ppg: " + str(ppg_a),
        "- btts: " + str(btts) + " | over25: " + str(over25),
        "LIVE:",
        "- state: " + state_name + " | minute: " + str(minute) + " | score: " + str(h_score) + "-" + str(a_score),
        "- stats: " + matched['live_stats_summary'],
        "- prematch_weight: " + str(prematch_weight) + " | live_weight: " + str(live_weight),
        "ALLOWED_DECISIONS:",
        json.dumps(allowed, ensure_ascii=False),
    ])

    default_payload = {
        "status_summary": "Veri eksik veya celiskili, pas gecildi.",
        "final_decision": "PAS GEC",
        "bankroll_percent": 0,
        "risk_score": 9,
        "confidence": 20,
        "reason_codes": ["LOW_DATA"]
    }
    ai_payload = default_payload
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config={"response_mime_type": "application/json", "response_json_schema": schema},
        )
        parsed = json.loads(response.text)
        if parsed.get('final_decision') not in allowed:
            parsed['status_summary'] = 'Model menu disina cikti, karar reddedildi.'
            parsed['final_decision'] = 'PAS GEC'
            parsed['reason_codes'] = ['CONFLICT']
        ai_payload = parsed
        log['ai_ok'] += 1
        if ai_payload['final_decision'] == 'PAS GEC':
            log['pas_gec'] += 1
    except Exception as e:
        print("AI hatasi: " + str(e))
        log['errors'] += 1

    matched['boss_ai_decision'] = (
        "DURUM: " + ai_payload['status_summary'] + "\n" +
        "KARAR: " + ai_payload['final_decision'] + "\n" +
        "KASA: %" + str(ai_payload['bankroll_percent']) +
        " | Risk: " + str(ai_payload['risk_score']) +
        "/10 | Guven: " + str(ai_payload['confidence']) + "/100"
    )
    print("OK: " + sm_home + " " + str(h_score) + "-" + str(a_score) + " " + sm_away + " -> " + ai_payload['final_decision'])
    time.sleep(1)

today_path.write_text(json.dumps({"data": fs_data}, ensure_ascii=False, indent=2), encoding='utf-8')
live_path.write_text(json.dumps(live_payload, ensure_ascii=False, indent=2), encoding='utf-8')

print("=== LOG ===")
print("Toplam: " + str(log['total']))
print("Eslesen: " + str(log['matched']))
print("Skor yazilan: " + str(log['scored']))
print("AI karar: " + str(log['ai_ok']))
print("PAS GEC: " + str(log['pas_gec']))
print("Hata: " + str(log['errors']))
print("Tamamlandi.")
