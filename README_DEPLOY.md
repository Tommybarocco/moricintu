
# Prophet Forecast API (FastAPI)

API web qui calcule une tendance avec Facebook Prophet à partir des données Binance.

## Déploiement rapide sur Render (gratuit)
1. Crée un repo GitHub avec ces 3 fichiers: `main.py`, `requirements.txt`, `Procfile`.
2. Va sur https://render.com -> New -> Web Service -> connecte ton repo.
3. Build command: `pip install -r requirements.txt`
   Start command: auto (pris du Procfile).
4. (Optionnel) Ajoute la variable d'env `API_TOKEN` pour sécuriser.
5. Déploie. Teste `GET /health` puis `GET /forecast?symbol=BTCUSDT&interval=1h&periods=24`.

## Utilisation depuis n8n (Cloud)
- Nœud **HTTP Request**: 
  - GET `https://<ton-app>.onrender.com/forecast`
  - Query: symbol=`{{$json.symbol}}`, interval=`{{$json.interval}}`, periods=`{{$items("Set Params")[0].json.prophetPeriods}}`
  - Headers si besoin: `x-api-token: <ton_token>`

Réponse JSON: `trend`, `score` (1..3), `delta`, `last_yhat`, etc.
