"""Mappa di rischio senza score: volatilità GARCH e probabilità di tocco.

Non usa il punteggio dealer-flow e non stima pesi. Due numeri, entrambi
funzione solo della volatilità point-in-time e (per i livelli) della distanza:

* range atteso a 5 giorni e P(range 5g > mediana trailing di 90 finestre),
  da un GARCH(1,1) sui rendimenti giornalieri (EWMA λ=0.94 se il fit non
  converge);
* P(tocco del livello entro 5 giorni | distanza in unità di σ_5g).

Scelta sulla probabilità di tocco
---------------------------------
Il backtest (10/11/2026) confronta due stimatori che dipendono solo da
``z = |ln(livello/ref)| / σ_5g``:

* A — principio di riflessione per un moto browniano senza drift,
  ``2 (1 − Φ(z))``. Formula chiusa, nessun campione.
* B — frequenza storica delle escursioni normalizzate (massimo rialzo e
  massimo ribasso di ogni finestra passata, messi nello stesso mucchio)
  che superano quello stesso ``z``. È la tabella di calibrazione empirica
  del backtest (Brier 0,086 e AUC 0,94 sui tocchi di max pain: l'AUC è
  della distanza in vol, quindi A e B ordinano allo stesso modo; cambia
  il Brier).

``touch_prob_5d`` è B quando ci sono almeno ``MIN_EXCURSIONS`` escursioni
già chiuse, altrimenti A. ``touch_prob_reflection`` è sempre A.
``scripts/verify_risk_map.py`` ricalcola entrambi in walk-forward.

Il range atteso è ``E[ln(H/L)] = σ_5g √(8/π)`` (range di un browniano
senza drift: per linearità ``E[max−min] = 2 σ √(2T/π)``). La probabilità
mostrata che il range superi la mediana a 90 giorni è la frequenza storica
dei range passati, misurati in unità del σ che era noto all'inizio di
ciascuna finestra: niente coefficienti. La sopravvivenza della legge
browniana ``1 − F(mediana / σ_5g)`` resta nel campo
``prob_range_gt_median_90d_brownian`` (sulle barre 2023–2026 discrimina,
ma è meno calibrata della frequenza). Sotto le 30 finestre chiuse si usa
quella formula, perché la tabella non c'è ancora.

Niente lookahead
----------------
Si usano solo barre giornaliere chiuse prima di ``asof`` (la barra del
giorno UTC corrente, ancora aperta, è esclusa). I parametri GARCH si
rifittano ogni 7 giorni sui rendimenti fino al giorno prima, come nel
backtest, e di nuovo sull'ultimo giorno. La mediana e la tabella empirica
contengono solo finestre già chiuse prima della decisione.
"""
from __future__ import annotations

import math
from collections import deque
from datetime import date, datetime, timezone
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

HORIZON_DAYS = 5
MEDIAN_LOOKBACK = 90
EWMA_LAMBDA = 0.94
MIN_RETURNS = 40
MIN_MEDIAN_WINDOWS = 20
MIN_EXCURSIONS = 60
REFIT_EVERY = 7
# Griglia di distanza (in σ_5g) usata dal verifier per i tocchi sintetici.
# Copre sia livelli vicini (max pain tipico) sia livelli lontani.
TOUCH_Z_GRID = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)

# CDF del range di un browniano standard su [0, 1] (deviazione del valore
# finale = 1). Ottenuta integrando la serie di riflessione della probabilità
# di restare in un intervallo, poi normalizzata: la quadratura cruda si
# ferma a ~0,996 e dopo la divisione E[R] = √(8/π) entro lo 0,1%.
# z = range / σ_orizzonte. Non è fittata sui prezzi.
_RANGE_Z = np.array(
    [
        0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40,
        1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 2.10, 2.20, 2.30, 2.40,
        2.50, 2.60, 2.70, 2.80, 2.90, 3.00, 3.10, 3.20, 3.30, 3.40,
        3.50, 3.60, 3.70, 3.80, 3.90, 4.00, 4.10, 4.20, 4.30, 4.40, 4.50,
    ]
)
_RANGE_F = np.array(
    [
        0.0000, 0.0000, 0.0007, 0.0060, 0.0242, 0.0635, 0.1259, 0.2071, 0.2994, 0.3949,
        0.4875, 0.5730, 0.6493, 0.7155, 0.7718, 0.8189, 0.8576, 0.8891, 0.9145, 0.9346,
        0.9505, 0.9628, 0.9724, 0.9797, 0.9852, 0.9893, 0.9923, 0.9946, 0.9962, 0.9973,
        0.9982, 0.9988, 0.9992, 0.9995, 0.9996, 0.9998, 0.9999, 0.9999, 1.0000, 1.0000,
        1.0000,
    ]
)

_RANGE_MEAN_FACTOR = math.sqrt(8.0 / math.pi)  # E[range] / σ_orizzonte


def reflection_touch_prob(z: float) -> float:
    """P(tocco entro l'orizzonte) = 2 (1 − Φ(z)), z = |ln(L/S)| / σ_orizzonte.

    Moto browniano senza drift. A z=0 il livello è il prezzo di riferimento
    e la probabilità è 1. Per z>0 il valore resta sotto 1.
    """
    if not math.isfinite(z):
        return 0.0
    if z <= 0:
        return 1.0
    if z >= 8:
        return 0.0
    return float(2.0 * (1.0 - norm.cdf(z)))


def range_cdf(z: float) -> float:
    """P(range / σ_orizzonte ≤ z) per un browniano senza drift."""
    if not math.isfinite(z) or z <= 0:
        return 0.0
    if z >= float(_RANGE_Z[-1]):
        return 1.0
    if z <= float(_RANGE_Z[0]):
        return float(_RANGE_F[0])
    return float(np.interp(z, _RANGE_Z, _RANGE_F))


def prob_range_exceeds(threshold_log: float, sigma_h: float) -> Optional[float]:
    """P(ln(H/L) > soglia) con soglia e σ nella stessa unità di log-prezzo."""
    if sigma_h is None or not math.isfinite(sigma_h) or sigma_h <= 0:
        return None
    if threshold_log is None or not math.isfinite(threshold_log):
        return None
    if threshold_log <= 0:
        return 1.0
    return float(1.0 - range_cdf(threshold_log / sigma_h))


def expected_log_range(sigma_h: float) -> Optional[float]:
    """E[ln(max/min)] sull'orizzonte. ``sigma_h`` è la deviazione del log-return a H giorni."""
    if sigma_h is None or not math.isfinite(sigma_h) or sigma_h <= 0:
        return None
    return float(sigma_h * _RANGE_MEAN_FACTOR)


# ─── Volatilità ──────────────────────────────────────────────────────────────


class _VolModel:
    """GARCH(1,1) con variance targeting, oppure EWMA di RiskMetrics."""

    def __init__(self, method: str, omega: float, alpha: float, beta: float,
                 ewma_lambda: float = EWMA_LAMBDA) -> None:
        self.method = method
        self.omega = float(omega)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.ewma_lambda = float(ewma_lambda)

    def as_dict(self) -> dict[str, Optional[float]]:
        if self.method == "ewma":
            return {"omega": None, "alpha": None, "beta": None, "lambda": self.ewma_lambda}
        return {
            "omega": self.omega,
            "alpha": self.alpha,
            "beta": self.beta,
            "lambda": None,
        }


def _ewma(var: float) -> _VolModel:
    return _VolModel("ewma", omega=0.0, alpha=1.0 - EWMA_LAMBDA, beta=EWMA_LAMBDA)


def fit_vol(returns: np.ndarray) -> _VolModel:
    """GARCH(1,1) gaussiano. EWMA λ=0.94 se il campione è corto o il fit fallisce.

    ``ω`` non è libero: variance targeting ``ω = Var(r) (1 − α − β)``, così
    il fit ha due parametri e non diverge su serie corte. α, β ≥ 0 e
    α+β < 0,995 (stazionarietà). Nessun input di score.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < MIN_RETURNS:
        return _ewma(float(np.var(r)) if len(r) else 1e-6)
    var = float(np.var(r))
    if not math.isfinite(var) or var <= 0:
        return _ewma(1e-8)

    def nll(theta: np.ndarray) -> float:
        alpha = float(theta[0])
        beta = float(theta[1])
        if alpha < 0 or beta < 0 or alpha + beta >= 0.995:
            return 1e12
        omega = var * (1.0 - alpha - beta)
        if omega <= 0:
            return 1e12
        v = var
        acc = 0.0
        for i, x in enumerate(r):
            if i >= 15 and v > 0:
                acc += math.log(v) + (x * x) / v
            v = omega + alpha * x * x + beta * v
            if v <= 1e-18 or not math.isfinite(v):
                return 1e12
        return 0.5 * acc

    try:
        res = minimize(
            nll,
            np.array([0.05, 0.90]),
            method="L-BFGS-B",
            bounds=[(1e-8, 0.40), (0.50, 0.989)],
            options={"maxiter": 80, "ftol": 1e-8},
        )
    except Exception:
        return _ewma(var)
    if not res.success:
        return _ewma(var)
    alpha, beta = float(res.x[0]), float(res.x[1])
    if alpha < 0 or beta < 0 or alpha + beta >= 0.995:
        return _ewma(var)
    omega = var * (1.0 - alpha - beta)
    if omega <= 0 or not math.isfinite(omega):
        return _ewma(var)
    return _VolModel("garch", omega, alpha, beta)


def _filter_variance(returns: np.ndarray, model: _VolModel) -> np.ndarray:
    """Varianza condizionale di ciascun rendimento (lunghezza = len(returns))."""
    r = np.asarray(returns, dtype=float)
    n = len(r)
    v = np.empty(n, dtype=float)
    if n == 0:
        return v
    var = float(np.var(r)) if n > 1 else float(r[0] ** 2)
    if not math.isfinite(var) or var <= 0:
        var = 1e-8
    if model.method == "ewma":
        lam = model.ewma_lambda
        v[0] = var
        for i in range(1, n):
            v[i] = lam * v[i - 1] + (1.0 - lam) * r[i - 1] * r[i - 1]
        return v
    v[0] = var
    for i in range(1, n):
        v[i] = model.omega + model.alpha * r[i - 1] * r[i - 1] + model.beta * v[i - 1]
    return v


def _forecast_sigma_h(last_var: float, last_return: float, model: _VolModel, horizon: int) -> float:
    """Deviazione standard del log-return sull'orizzonte (somma delle varianze attese)."""
    horizon = int(horizon)
    if model.method == "ewma":
        lam = model.ewma_lambda
        nxt = lam * last_var + (1.0 - lam) * last_return * last_return
        return float(math.sqrt(max(nxt, 0.0) * horizon))
    nxt = model.omega + model.alpha * last_return * last_return + model.beta * last_var
    persist = model.alpha + model.beta
    uncond = model.omega / (1.0 - persist)
    total = 0.0
    v = max(nxt, 0.0)
    for _ in range(horizon):
        total += v
        v = uncond + persist * (v - uncond)
        if v < 0:
            v = 0.0
    return float(math.sqrt(total))


# ─── Barre ───────────────────────────────────────────────────────────────────


def _as_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def prepare_bars(bars: pd.DataFrame | Sequence[dict], *, asof: datetime | date | None = None
                 ) -> pd.DataFrame:
    """Barre OHLC chiuse strettamente prima di ``asof``.

    Scarta ``partial=1``, le date ``>= asof`` (la sessione UTC del giorno
    di previsione è ancora aperta) e i prezzi non positivi. Se ``asof`` è
    assente, scarta l'ultima barra solo quando ``close_time`` è nel futuro.
    """
    df = bars.copy() if isinstance(bars, pd.DataFrame) else pd.DataFrame(list(bars))
    if df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close"])
    df.columns = [str(c).lower() for c in df.columns]
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"manca la colonna {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = [_as_date(v) for v in df["date"]]
    if "partial" in df.columns:
        flag = pd.to_numeric(df["partial"], errors="coerce").fillna(0).astype(int)
        df = df.loc[flag == 0]
    if asof is not None:
        cut = asof.date() if isinstance(asof, datetime) else asof
        df = df.loc[df["date"] < cut]
    elif "close_time" in df.columns:
        now = datetime.now(timezone.utc)
        closed = []
        for value in df["close_time"]:
            if value is None or (isinstance(value, float) and math.isnan(value)):
                closed.append(True)
                continue
            ts = value
            if not isinstance(ts, datetime):
                ts = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            closed.append(ts <= now)
        df = df.loc[np.array(closed)]
    df = df.dropna(subset=["high", "low", "close"])
    df = df.loc[(df["close"] > 0) & (df["high"] > 0) & (df["low"] > 0)]
    df = df.sort_values("date").drop_duplicates("date", keep="last")
    return df.reset_index(drop=True)


# api.binance.com risponde 451 da alcune reti (GitHub, questa VM). Lo stesso
# libro BTCUSDT è pubblicato su data-api.binance.vision.
_BINANCE_KLINE_HOSTS = (
    "https://api.binance.com",
    "https://data-api.binance.vision",
)


def fetch_binance_daily(
    symbol: str = "BTCUSDT",
    start: str = "2023-01-01",
    *,
    timeout: float = 20.0,
) -> pd.DataFrame:
    """Kline giornaliere Binance (UTC). Include la barra del giorno in corso: va filtrata."""
    import requests

    start_ms = int(datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp() * 1000)
    rows: list[list] = []
    host = _kline_host(symbol, start_ms, timeout)
    url = f"{host}/api/v3/klines"
    while True:
        resp = requests.get(
            url,
            params={"symbol": symbol, "interval": "1d", "startTime": start_ms, "limit": 1000},
            timeout=timeout,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        last_open = int(batch[-1][0])
        nxt = last_open + 1
        if nxt <= start_ms or len(batch) < 1000:
            break
        start_ms = nxt
    recs = []
    for k in rows:
        open_dt = datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc)
        close_dt = datetime.fromtimestamp(k[6] / 1000, tz=timezone.utc)
        recs.append({
            "date": open_dt.date(),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "close_time": close_dt,
        })
    return pd.DataFrame(recs)


def _kline_host(symbol: str, start_ms: int, timeout: float) -> str:
    """Primo host Binance che risponde. 451/403 passano al successivo."""
    import requests

    last_error: Exception | None = None
    for host in _BINANCE_KLINE_HOSTS:
        try:
            resp = requests.get(
                f"{host}/api/v3/klines",
                params={"symbol": symbol, "interval": "1d", "startTime": start_ms, "limit": 1},
                timeout=timeout,
            )
        except requests.RequestException as exc:
            last_error = exc
            continue
        if resp.status_code in (401, 403, 451):
            last_error = requests.HTTPError(f"{resp.status_code} da {host}", response=resp)
            continue
        resp.raise_for_status()
        return host
    if last_error is not None:
        raise last_error
    raise RuntimeError("nessun host Binance configurato")


# ─── Walk-forward ────────────────────────────────────────────────────────────


def _num(value: Any, ndigits: int | None = None) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    if ndigits is None:
        return out
    return round(out, ndigits)


def _median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=float)))


def walk_forward(
    bars: pd.DataFrame | Sequence[dict],
    *,
    asof: datetime | date | None = None,
    horizon: int = HORIZON_DAYS,
    refit_every: int = REFIT_EVERY,
    median_lookback: int = MEDIAN_LOOKBACK,
    z_grid: Sequence[float] = (),
    params_mode: str = "refit",
) -> dict[str, Any]:
    """Una decisione per ogni origine con abbastanza storia.

    ``range_rows`` ha l'esito solo se la finestra futura è già nel frame
    (il verifier). ``latest`` è la decisione sull'ultima barra chiusa, senza
    esito. ``excursions`` sono le escursioni ``/σ`` delle finestre già chiuse
    prima di ``latest`` (sopra e sotto, insieme).

    ``params_mode="refit"`` ristima il GARCH ogni ``refit_every`` origini, solo
    sui rendimenti già noti (è il verifier). ``params_mode="once"`` stima una
    volta sull'intero frame già chiuso e riusa quei parametri: è il predict
    live, che non può permettersi centinaia di fit a ogni cohort.
    """
    df = prepare_bars(bars, asof=asof)
    empty: dict[str, Any] = {
        "latest": None,
        "range_rows": [],
        "touch_rows": [],
        "excursions": np.array([], dtype=float),
        "range_ratios": np.array([], dtype=float),
    }
    if len(df) < horizon + 5:
        return empty

    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    dates = list(df["date"])
    log_ret = np.diff(np.log(close))
    n = len(close)
    horizon = int(horizon)
    start = max(horizon + 1, MIN_RETURNS + 1)
    if start > n:
        return empty

    model: _VolModel | None = None
    fit_at = -10**9
    # "once": un solo fit sui rendimenti del frame (già tagliato a asof).
    frozen: _VolModel | None = None
    if params_mode == "once" and len(log_ret) >= MIN_RETURNS:
        frozen = fit_vol(log_ret)
    ratios: list[float] = []
    excursions: list[float] = []
    pending: deque[tuple[int, float, float, float]] = deque()
    range_rows: list[dict[str, Any]] = []
    touch_rows: list[dict[str, Any]] = []
    latest: dict[str, Any] | None = None
    grid = tuple(float(z) for z in z_grid)

    def _release(origin: int) -> None:
        while pending and pending[0][0] <= origin:
            _, ratio, up_z, down_z = pending.popleft()
            ratios.append(ratio)
            excursions.append(up_z)
            excursions.append(down_z)

    for origin in range(start, n + 1):
        _release(origin)
        r = log_ret[: origin - 1]
        if len(r) < MIN_RETURNS:
            continue
        if frozen is not None:
            model = frozen
        elif model is None or origin - fit_at >= refit_every or origin == n:
            model = fit_vol(r)
            fit_at = origin
        assert model is not None
        var = _filter_variance(r, model)
        sigma_h = _forecast_sigma_h(float(var[-1]), float(r[-1]), model, horizon)
        med_values: list[float] = []
        e_last = origin - 1
        e_min = horizon - 1
        for e in range(e_last, e_min - 1, -1):
            sl = slice(e - horizon + 1, e + 1)
            hh = float(high[sl].max())
            ll = float(low[sl].min())
            if ll <= 0 or hh < ll:
                continue
            med_values.append(math.log(hh / ll))
            if len(med_values) >= median_lookback:
                break
        median = _median(med_values) if len(med_values) >= MIN_MEDIAN_WINDOWS else None
        p_model = prob_range_exceeds(median, sigma_h) if median is not None else None
        p_emp: Optional[float] = None
        if median is not None and sigma_h > 0 and len(ratios) >= MIN_EXCURSIONS // 2:
            z_thr = median / sigma_h
            p_emp = float(np.mean(np.asarray(ratios) >= z_thr))

        record = {
            "origin_date": dates[origin].isoformat() if origin < n else None,
            "last_bar_date": dates[origin - 1].isoformat(),
            "ref_close": float(close[origin - 1]),
            "sigma_h": sigma_h,
            "vol_method": model.method,
            "vol_params": model.as_dict(),
            "n_returns": int(len(r)),
            "median_log_range": median,
            "n_median": len(med_values),
            "prob_range_model": p_model,
            "prob_range_empirical": p_emp,
            "n_range_samples": len(ratios),
            "n_excursions": len(excursions),
        }
        latest = record

        if origin + horizon <= n and sigma_h > 0 and median is not None and p_model is not None:
            sl = slice(origin, origin + horizon)
            hh = float(high[sl].max())
            ll = float(low[sl].min())
            ref = float(close[origin - 1])
            realized = math.log(hh / ll) if ll > 0 and hh >= ll else None
            if realized is not None and ref > 0:
                y = 1 if realized > median else 0
                up_z = max(0.0, math.log(hh / ref)) / sigma_h
                down_z = max(0.0, math.log(ref / ll)) / sigma_h
                ratio = realized / sigma_h
                range_rows.append({
                    **record,
                    "y_range": y,
                    "realized_log_range": realized,
                })
                for z in grid:
                    p_ref = reflection_touch_prob(z)
                    p_hist = (
                        float(np.mean(np.asarray(excursions) >= z))
                        if len(excursions) >= MIN_EXCURSIONS else None
                    )
                    p_used = p_hist if p_hist is not None else p_ref
                    for side, hit in (("above", up_z >= z), ("below", down_z >= z)):
                        touch_rows.append({
                            "origin_date": record["origin_date"],
                            "last_bar_date": record["last_bar_date"],
                            "n_returns": record["n_returns"],
                            "z": z,
                            "side": side,
                            "y": 1 if hit else 0,
                            "p_reflection": p_ref,
                            "p_empirical": p_hist,
                            "p_used": p_used,
                            "touch_method": (
                                "empirical_excursion" if p_hist is not None else "reflection"
                            ),
                        })
                pending.append((origin + horizon, ratio, up_z, down_z))

    return {
        "latest": latest,
        "range_rows": range_rows,
        "touch_rows": touch_rows,
        "excursions": np.asarray(excursions, dtype=float),
        "range_ratios": np.asarray(ratios, dtype=float),
    }


def _level_view(ref: float, level: dict[str, Any], sigma_h: Optional[float],
                excursions: np.ndarray) -> Optional[dict[str, Any]]:
    price = _num(level.get("price"))
    if price is None or price <= 0 or ref <= 0:
        return None
    dist_pct = (price / ref - 1.0) * 100.0
    log_abs = abs(math.log(price / ref))
    z = (log_abs / sigma_h) if sigma_h and sigma_h > 0 else None
    p_ref = reflection_touch_prob(z) if z is not None else None
    p_emp = None
    if z is not None and len(excursions) >= MIN_EXCURSIONS:
        p_emp = float(np.mean(excursions >= z))
    if p_emp is not None:
        p_used, method = p_emp, "empirical_excursion"
    else:
        p_used, method = p_ref, "reflection"
    out: dict[str, Any] = {
        "name": level.get("name"),
        "price": round(price, 2),
        "side": "above" if price >= ref else "below",
        "distance_pct": round(dist_pct, 4),
        "distance_vol": round(z, 4) if z is not None else None,
        "touch_prob_5d": round(p_used, 6) if p_used is not None else None,
        "touch_prob_reflection": round(p_ref, 6) if p_ref is not None else None,
        "touch_method": method if p_used is not None else None,
    }
    for key in ("barrier_id", "barrier_type", "issuer"):
        if level.get(key) is not None:
            out[key] = level[key]
    return out


def build_risk_map(
    bars: pd.DataFrame | Sequence[dict],
    *,
    ref_price: float,
    levels: Sequence[dict[str, Any]] | None = None,
    asof: datetime | date | None = None,
    horizon: int = HORIZON_DAYS,
    source: str = "binance:BTCUSDT:1d",
) -> dict[str, Any]:
    """Mappa di rischio alla chiusura dell'ultima barra disponibile prima di ``asof``.

    ``ref_price`` è lo spot della decisione (Deribit nel predict live), non
    necessariamente il close Binance: le distanze sono misurate da lì, la
    volatilità dalle barre. Nessun campo di score.
    """
    ref = float(ref_price)
    if not math.isfinite(ref) or ref <= 0:
        raise ValueError("ref_price deve essere positivo")
    wf = walk_forward(bars, asof=asof, horizon=horizon, z_grid=(), params_mode="once")
    latest = wf["latest"] or {}
    sigma_h = latest.get("sigma_h")
    median = latest.get("median_log_range")
    exp_range = expected_log_range(sigma_h) if sigma_h else None
    excursions = wf["excursions"]
    level_rows = []
    methods = set()
    for level in levels or []:
        view = _level_view(ref, level, sigma_h, excursions)
        if view is None:
            continue
        level_rows.append(view)
        if view.get("touch_method"):
            methods.add(view["touch_method"])
    if not methods:
        touch_method = (
            "empirical_excursion" if len(excursions) >= MIN_EXCURSIONS else "reflection"
        )
    elif methods == {"empirical_excursion"}:
        touch_method = "empirical_excursion"
    elif methods == {"reflection"}:
        touch_method = "reflection"
    else:
        touch_method = "mixed"

    asof_iso = None
    if isinstance(asof, datetime):
        if asof.tzinfo is None:
            asof = asof.replace(tzinfo=timezone.utc)
        asof_iso = asof.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    elif isinstance(asof, date):
        asof_iso = datetime(asof.year, asof.month, asof.day, tzinfo=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

    def _px(log_value: Optional[float]) -> tuple[Optional[float], Optional[float]]:
        if log_value is None or not math.isfinite(log_value):
            return None, None
        return round(ref * log_value, 2), round(100.0 * log_value, 4)

    move_usd, move_pct = _px(sigma_h)
    range_usd, range_pct = _px(exp_range)
    _, median_pct = _px(median)
    p_model = latest.get("prob_range_model")
    p_emp = latest.get("prob_range_empirical")
    if p_emp is not None:
        p_show, p_method = p_emp, "empirical_range"
    else:
        p_show, p_method = p_model, "brownian_range_cdf"
    return {
        "asof": asof_iso,
        "horizon_days": int(horizon),
        "source": source,
        "last_bar_date": latest.get("last_bar_date"),
        "n_returns": latest.get("n_returns"),
        "ref_price": round(ref, 2),
        "vol_method": latest.get("vol_method"),
        "vol_params": latest.get("vol_params"),
        "sigma_5d_log": round(sigma_h, 6) if sigma_h else None,
        "move_1sigma_5d_usd": move_usd,
        "move_1sigma_5d_pct": move_pct,
        "expected_range_5d_log": round(exp_range, 6) if exp_range else None,
        "expected_range_5d_usd": range_usd,
        "expected_range_5d_pct": range_pct,
        "range_median_90d_log": round(median, 6) if median else None,
        "range_median_90d_pct": median_pct,
        "prob_range_gt_median_90d": round(p_show, 6) if p_show is not None else None,
        "prob_range_method": p_method if p_show is not None else None,
        "prob_range_gt_median_90d_brownian": round(p_model, 6) if p_model is not None else None,
        "prob_range_gt_median_90d_empirical": round(p_emp, 6) if p_emp is not None else None,
        "touch_method": touch_method,
        "n_excursions": int(len(excursions)),
        "n_median": latest.get("n_median"),
        "levels": level_rows,
    }


def gex_levels(snapshot: Any) -> list[dict[str, Any]]:
    """Put wall, call wall, gamma flip, max pain da uno snapshot GEX (o dict)."""
    spec = (
        ("put_wall", "put_wall"),
        ("call_wall", "call_wall"),
        ("gamma_flip", "gamma_flip_price"),
        ("max_pain", "max_pain"),
    )
    out = []
    for name, attr in spec:
        raw = snapshot.get(attr) if isinstance(snapshot, dict) else getattr(snapshot, attr, None)
        # dict-like snapshot può avere il nome corto
        if raw is None and isinstance(snapshot, dict) and attr == "gamma_flip_price":
            raw = snapshot.get("gamma_flip")
        price = _num(raw)
        if price is not None and price > 0:
            out.append({"name": name, "price": price})
    return out


def barrier_levels(barriers: Iterable[dict] | None) -> list[dict[str, Any]]:
    """Barriere EDGAR con prezzo BTC già noto. Senza prezzo non entrano nella mappa."""
    out = []
    for b in barriers or []:
        price = _num(b.get("level_price_btc") if isinstance(b, dict) else getattr(b, "level_price_btc", None))
        if price is None or price <= 0:
            continue
        if isinstance(b, dict):
            out.append({
                "name": "barrier",
                "price": price,
                "barrier_id": b.get("id"),
                "barrier_type": b.get("barrier_type"),
                "issuer": b.get("issuer"),
            })
        else:
            out.append({"name": "barrier", "price": price})
    return out


def levels_pit(
    snapshot: Any,
    barriers: Iterable[dict] | None,
    *,
    ref_price: float,
    timestamp: str,
) -> dict[str, Any]:
    """Tutti i livelli point-in-time, anche quelli senza prezzo (restano null)."""

    def _get(attr: str, alt: str | None = None) -> Optional[float]:
        if isinstance(snapshot, dict):
            raw = snapshot.get(attr)
            if raw is None and alt:
                raw = snapshot.get(alt)
        else:
            raw = getattr(snapshot, attr, None)
        return _num(raw)

    gex = _get("total_net_gex", "total_gex")
    if gex is None:
        sign = None
    elif gex > 0:
        sign = 1
    elif gex < 0:
        sign = -1
    else:
        sign = 0
    barrier_rows = []
    for b in barriers or []:
        if not isinstance(b, dict):
            continue
        barrier_rows.append({
            "id": b.get("id"),
            "barrier_type": b.get("barrier_type"),
            "issuer": b.get("issuer"),
            "level_price_btc": _num(b.get("level_price_btc"), 2),
            "level_price_ibit": _num(b.get("level_price_ibit"), 6),
            "notional_usd": _num(b.get("notional_usd"), 2),
            "status": b.get("status") or "active",
        })
    return {
        "timestamp": timestamp,
        "ref_price": round(float(ref_price), 2),
        "put_wall": _num(_get("put_wall"), 2),
        "call_wall": _num(_get("call_wall"), 2),
        "gamma_flip": _num(_get("gamma_flip_price", "gamma_flip"), 2),
        "max_pain": _num(_get("max_pain"), 2),
        "total_gex": _num(gex, 2),
        "gex_sign": sign,
        "barriers": barrier_rows,
    }


# ─── Metriche (verifier e test) ──────────────────────────────────────────────


def brier_score(y: Sequence[float], p: Sequence[float]) -> Optional[float]:
    yy = np.asarray(y, dtype=float)
    pp = np.asarray(p, dtype=float)
    mask = np.isfinite(yy) & np.isfinite(pp)
    if int(mask.sum()) == 0:
        return None
    err = pp[mask] - yy[mask]
    return float(np.mean(err * err))


def roc_auc(y: Sequence[float], p: Sequence[float]) -> Optional[float]:
    """AUC di Mann–Whitney con rango medio sui pareggi."""
    yy = np.asarray(y, dtype=float)
    pp = np.asarray(p, dtype=float)
    mask = np.isfinite(yy) & np.isfinite(pp)
    yy, pp = yy[mask], pp[mask]
    n_pos = int(np.sum(yy == 1))
    n_neg = int(np.sum(yy == 0))
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(pp, kind="mergesort")
    sorted_p = pp[order]
    ranks = np.empty(len(pp), dtype=float)
    i = 0
    while i < len(pp):
        j = i
        while j + 1 < len(pp) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * ((i + 1) + (j + 1))
        i = j + 1
    sum_pos = float(ranks[yy == 1].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def expanding_constant_brier(y: Sequence[float]) -> Optional[float]:
    """Brier di una probabilità costante = frequenza delle osservazioni precedenti."""
    yy = np.asarray(y, dtype=float)
    yy = yy[np.isfinite(yy)]
    if len(yy) == 0:
        return None
    p = np.empty(len(yy), dtype=float)
    p[0] = 0.5
    if len(yy) > 1:
        csum = np.cumsum(yy)
        p[1:] = csum[:-1] / np.arange(1, len(yy))
    err = p - yy
    return float(np.mean(err * err))


def calibration_table(y: Sequence[float], p: Sequence[float],
                      edges: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0001)
                      ) -> list[dict[str, Any]]:
    yy = np.asarray(y, dtype=float)
    pp = np.asarray(p, dtype=float)
    mask = np.isfinite(yy) & np.isfinite(pp)
    yy, pp = yy[mask], pp[mask]
    rows = []
    bounds = list(edges)
    for lo, hi in zip(bounds, bounds[1:]):
        sel = (pp >= lo) & (pp < hi)
        n = int(sel.sum())
        rows.append({
            "bin": f"{lo:.2f}-{min(hi, 1):.2f}",
            "n": n,
            "p_mean": round(float(pp[sel].mean()), 4) if n else None,
            "freq": round(float(yy[sel].mean()), 4) if n else None,
        })
    return rows


def score_probs(y: Sequence[float], p: Sequence[float]) -> dict[str, Any]:
    yy = np.asarray(y, dtype=float)
    pp = np.asarray(p, dtype=float)
    mask = np.isfinite(yy) & np.isfinite(pp)
    yy, pp = yy[mask], pp[mask]
    return {
        "n": int(len(yy)),
        "base_rate": round(float(yy.mean()), 4) if len(yy) else None,
        "brier": None if len(yy) == 0 else round(float(np.mean((pp - yy) ** 2)), 4),
        "brier_constant_expanding": (
            None if len(yy) == 0 else round(expanding_constant_brier(yy) or 0.0, 4)
        ),
        "auc": None if len(yy) == 0 or roc_auc(yy, pp) is None else round(roc_auc(yy, pp), 4),
        "calibration": calibration_table(yy, pp),
    }


def evaluate_walk_forward(wf: dict[str, Any]) -> dict[str, Any]:
    """Brier/AUC del modello di range e dei tocchi, contro la costante espandente."""
    rows = wf.get("range_rows") or []
    y = [r["y_range"] for r in rows]
    p_model = [r["prob_range_model"] for r in rows]
    p_emp = [r["prob_range_empirical"] if r["prob_range_empirical"] is not None else np.nan
             for r in rows]
    touch = wf.get("touch_rows") or []
    out: dict[str, Any] = {
        "range_model": score_probs(y, p_model),
        "range_empirical": score_probs(y, p_emp),
    }
    if touch:
        out["touch_used"] = score_probs([t["y"] for t in touch], [t["p_used"] for t in touch])
        out["touch_reflection"] = score_probs(
            [t["y"] for t in touch], [t["p_reflection"] for t in touch],
        )
        out["touch_empirical"] = score_probs(
            [t["y"] for t in touch],
            [t["p_empirical"] if t["p_empirical"] is not None else np.nan for t in touch],
        )
        near = [t for t in touch if t["z"] <= 1.0]
        out["touch_used_near_z_le_1"] = score_probs(
            [t["y"] for t in near], [t["p_used"] for t in near],
        )
        out["touch_reflection_near_z_le_1"] = score_probs(
            [t["y"] for t in near], [t["p_reflection"] for t in near],
        )
    else:
        out["touch_used"] = score_probs([], [])
    return out


def decision_asof_for(day: date) -> datetime:
    """07:30 UTC del giorno ``day``: le barre di quel giorno non sono ancora chiuse."""
    return datetime(day.year, day.month, day.day, 7, 30, tzinfo=timezone.utc)
