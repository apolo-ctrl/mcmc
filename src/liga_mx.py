# -*- coding: utf-8 -*-
"""
==========================================================================================
 PREDICCIÓN DE MARCADORES - LIGA MX (clubes)
==========================================================================================
 Mismo modelo que `prediccion_selecciones.py` (Dixon-Coles + MCMC Bayesiano + XGBoost),
 pero con datos de la Liga MX en vez de selecciones nacionales.

 Solo cambia la FUENTE DE DATOS y la carga; el resto de funciones (Dixon-Coles, modelo
 bayesiano, XGBoost, matriz de marcadores, accuracy, visualización) se REUTILIZAN
 importándolas desde prediccion_selecciones.py.

 Fuente de datos: football-data.co.uk (CSV de México, con goles y cuotas)
   https://www.football-data.co.uk/new/MEX.csv
 Columnas relevantes: Date, Home, Away, HG (goles local), AG (goles visitante), Res

 Alternativa (formato footballcsv en GitHub):
   https://github.com/footballcsv/mexico  ->  columnas: Round, Date, Team 1, FT, Team 2
==========================================================================================
"""

# =========================================================================================
# 1-2. LIBRERÍAS
# =========================================================================================
#   pip install pandas numpy scipy xgboost scikit-learn pymc arviz matplotlib
import io
import urllib.request
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# Reutilizamos TODO el modelo del script de selecciones.
# (Al ejecutarse con `python src/liga_mx.py`, la carpeta src/ está en el path.)
from prediccion_selecciones import (
    ajustar_dixon_coles,
    construir_variables_xgb,
    entrenar_regresores_xgb,
    entrenar_bayesiano,
    predecir_bayesiano,
    predecir_xgboost,
    _resumen_matriz,
    imprimir_top,
    visualizar,
    accuracy_bayes,
    accuracy_xgb,
)

# =========================================================================================
# 3. CONFIGURACIÓN
# =========================================================================================
DATA_URL_MX = "https://www.football-data.co.uk/new/MEX.csv"

# Partido a predecir (usa los nombres EXACTOS del dataset; si fallas, el script te
# imprime la lista de equipos válidos).
HOME_TEAM = "America"
AWAY_TEAM = "Guadalajara"
NEUTRAL   = False          # en liga normalmente hay localía -> False
MAX_GOALS = 8
TEST_FRACTION = 0.15
ANIO_MIN = 2018


# =========================================================================================
# 4. DESCARGA Y LIMPIEZA DE DATOS (Liga MX)
# =========================================================================================
def _descargar_csv(url):
    """Descarga un CSV usando un User-Agent de navegador (football-data.co.uk bloquea
    las peticiones automáticas sin cabecera y devuelve 403)."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read()
    return pd.read_csv(io.BytesIO(raw))


def cargar_datos_mx(url=DATA_URL_MX, anio_min=ANIO_MIN):
    """Descarga, limpia y filtra los datos de la Liga MX al formato que espera el modelo.

    El modelo necesita estas columnas: date, home_team, away_team, home_score,
    away_score, neutral, result.
    """
    df = _descargar_csv(url)

    # --- Mapeo de columnas de football-data.co.uk al formato del modelo ---
    columnas = {
        "Home": "home_team",
        "Away": "away_team",
        "HG": "home_score",   # Full Time Home Goals
        "AG": "away_score",   # Full Time Away Goals
    }
    faltan = [c for c in columnas if c not in df.columns]
    if faltan:
        raise SystemExit(
            "El CSV no tiene las columnas esperadas "
            f"{list(columnas.keys())}. Columnas encontradas: {list(df.columns)}.\n"
            "Revisa la URL/fuente o ajusta el mapeo en cargar_datos_mx()."
        )
    df = df.rename(columns=columnas)

    # --- Fechas (formato dd/mm/yyyy en football-data) ---
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    # --- Limpieza ---
    df = df.dropna(subset=["date", "home_score", "away_score", "home_team", "away_team"])
    df = df[df["date"].dt.year >= anio_min].copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["home_team"] = df["home_team"].astype(str).str.strip()
    df["away_team"] = df["away_team"].astype(str).str.strip()

    # --- En liga siempre hay localía (no hay sedes neutrales) ---
    df["neutral"] = False

    # --- Resultado 1X2 (para medir accuracy) ---
    def res(r):
        if r["home_score"] > r["away_score"]:
            return "H"
        if r["home_score"] < r["away_score"]:
            return "A"
        return "D"
    df["result"] = df.apply(res, axis=1)

    df = df.sort_values("date").reset_index(drop=True)
    print(f"[datos] Partidos Liga MX cargados (>= {anio_min}): {len(df)}")
    print(f"[datos] Rango de fechas: {df['date'].min().date()} -> {df['date'].max().date()}")
    equipos = pd.unique(df[["home_team", "away_team"]].values.ravel())
    print(f"[datos] Equipos distintos: {equipos.size}")
    return df


# =========================================================================================
# 5. EJECUTAR (mismo flujo que selecciones)
# =========================================================================================
def main():
    df = cargar_datos_mx()

    # Split temporal train/test
    corte = int(len(df) * (1 - TEST_FRACTION))
    df_train = df.iloc[:corte].copy()
    df_test = df.iloc[corte:].copy()
    print(f"[split] train={len(df_train)}  test={len(df_test)} "
          f"(desde {df_test['date'].min().date()})")

    # Validación de nombres de equipo
    equipos = set(pd.unique(df_train[["home_team", "away_team"]].values.ravel()))
    if HOME_TEAM not in equipos or AWAY_TEAM not in equipos:
        raise SystemExit(
            "Configura HOME_TEAM/AWAY_TEAM con equipos válidos.\n"
            f"Equipos disponibles: {sorted(equipos)}"
        )

    # Dixon-Coles (variables para XGBoost)
    dc = ajustar_dixon_coles(df_train, xi=0.0)
    print("\n[dixon-coles] Fuerzas de los equipos configurados:")
    print(dc["strengths"].loc[[HOME_TEAM, AWAY_TEAM]].round(3))

    # XGBoost
    X, y_home, y_away = construir_variables_xgb(df_train, dc)
    modelos_xgb = entrenar_regresores_xgb(X, y_home, y_away)

    # MCMC Bayesiano
    modelo_bayes = entrenar_bayesiano(df_train, draws=1000, tune=1000, chains=2)

    # Predicciones
    print("\n" + "=" * 70)
    print(f"PREDICCIÓN LIGA MX: {HOME_TEAM} vs {AWAY_TEAM} (neutral={NEUTRAL})")
    print("=" * 70)
    M_bayes = predecir_bayesiano(modelo_bayes, HOME_TEAM, AWAY_TEAM, NEUTRAL, MAX_GOALS)
    M_xgb, lam_h, lam_a = predecir_xgboost(modelos_xgb, dc, HOME_TEAM, AWAY_TEAM,
                                           NEUTRAL, MAX_GOALS)

    for nombre, M in [("MCMC Bayesiano", M_bayes), ("XGBoost híbrido", M_xgb)]:
        ph, pd_, pa, (bi, bj, bp) = _resumen_matriz(M)
        print(f"\n[{nombre}]")
        print(f"  1X2 -> {HOME_TEAM}: {ph*100:5.1f}% | Empate: {pd_*100:5.1f}% | "
              f"{AWAY_TEAM}: {pa*100:5.1f}%")
        print(f"  Marcador más probable: {bi}-{bj} ({bp*100:.1f}%)")
        imprimir_top(M, HOME_TEAM, AWAY_TEAM, n=10)

    # Visualización
    fig1 = visualizar(M_bayes, HOME_TEAM, AWAY_TEAM, "Liga MX — MCMC Bayesiano")
    fig2 = visualizar(M_xgb, HOME_TEAM, AWAY_TEAM, "Liga MX — XGBoost híbrido",
                      lam=(lam_h, lam_a))
    fig1.savefig("ligamx_bayes.png", dpi=120)
    fig2.savefig("ligamx_xgboost.png", dpi=120)
    print("\n[plot] Guardado: ligamx_bayes.png, ligamx_xgboost.png")

    # Accuracy
    print("\n" + "=" * 70)
    print("ACCURACY (1X2) SOBRE EL CONJUNTO DE TEST")
    print("=" * 70)
    acc_b, n_b = accuracy_bayes(modelo_bayes, df_test, MAX_GOALS)
    acc_x, n_x = accuracy_xgb(modelos_xgb, dc, df_test, MAX_GOALS)
    base = (df_test["result"] == "H").mean()
    print(f"  Baseline (siempre local):  {base*100:5.2f}%")
    print(f"  MCMC Bayesiano:            {acc_b*100:5.2f}%   (n={n_b})")
    print(f"  XGBoost híbrido:           {acc_x*100:5.2f}%   (n={n_x})")


if __name__ == "__main__":
    main()
