import requests
import time

# API ANAHTARINI BURAYA GIR
API_KEY = "CpJnwdzG4RHQIs37bRQehp2iET5I7fQLoOCUMkT7qxNF78FABDIciJ7ka6o6"
BASE_URL = "https://api.sportmonks.com/v3/football/fixtures"

includes = [
    "participants", "scores", "events", "lineups", "odds", "statistics",
    "predictions", "trends", "pressure", "standings", "topscorers",
    "tvstations", "sidelined", "referees", "comments", "venue", "coaches",
    "periods", "league", "season", "form", "groupstandings", "news",
    "metadata", "state", "weatherReport", "probabilities"
]

results = []

print(f"Test baslatildi. Toplam {len(includes)} parametre sorgulanacak.")
print("Her sorgu arasi 10 saniye beklenecek. Lutfen bekleyin...")

for inc in includes:
    url = f"{BASE_URL}?api_token={API_KEY}&include={inc}"
    try:
        response = requests.get(url, timeout=15)
        status = f"[{response.status_code}] {inc}"
        print(status)
        results.append(status)
    except Exception as e:
        error_msg = f"[HATA] {inc}: Baglanti Sorunu"
        print(error_msg)
        results.append(error_msg)
    
    # Sunucu senkronizasyonu icin 10 saniye bekleme
    time.sleep(10)

with open("kesin_sonuc.txt", "w") as f:
    f.write("\n".join(results))

print("\nIslem tamamlandi. 'kesin_sonuc.txt' dosyasini buraya yapistir.")
