# -*- coding: utf-8 -*-
"""
==========================================================================================
 PREDICCIÓN DE MARCADORES DE SELECCIONES NACIONALES
==========================================================================================
 Dataset : martj42/international_results (results.csv)
 Pipeline:
   1) Limpieza y filtro de datos (>= 2018)
   2) Dixon-Coles  -> fuerza ofensiva / defensiva (variables para XGBoost)
   3) Modelo 1: MCMC Bayesiano jerárquico (Baio-Blangiardo, PyMC)
   4) Modelo 2: XGBoost híbrido (Groll) -> 2 regresores Poisson + estructura de Poisson
 Salida:
   - Matriz de probabilidad de cada marcador posible (heatmap)
   - Probabilidades 1X2 y marcador más probable
   - Accuracy de ambos modelos sobre un conjunto de test
==========================================================================================

Este archivo está pensado para ejecutarse de corrido (python prediccion_selecciones.py)
o para copiarse por bloques en un notebook de Google Colab. Cada sección está marcada
con un encabezado "# === N. ... ===" que coincide con el guion solicitado.
"""

# =========================================================================================
# 1. INSTALAR LIBRERÍAS
# =========================================================================================
# En Google Colab / entorno nuevo, descomenta la siguiente línea:
#
#   !pip install pandas numpy scipy xgboost scikit-learn pymc arviz matplotlib
#
# (En local: pip install pandas numpy scipy xgboost scikit-learn pymc arviz matplotlib)
#
# Nota: la API de scikit-learn de XGBoost (XGBRegressor) REQUIERE scikit-learn instalado.


# =========================================================================================
# 2. IMPORTAR LIBRERÍAS
# =========================================================================================
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

import xgboost as xgb

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# PyMC se importa dentro de la función del modelo bayesiano para que el resto del
# script funcione aunque PyMC no esté instalado.

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# =========================================================================================
# 3. DESCARGAR EL DATASET DEL REPOSITORIO
# =========================================================================================
DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"


def cargar_datos(url=DATA_URL, anio_min=2018):
    """Descarga, limpia y filtra el dataset de resultados internacionales.

    Limpieza:
      - parseo de fechas y filtro >= anio_min
      - eliminación de filas con goles nulos
      - tipado de goles a entero
      - cálculo de columnas auxiliares (año, resultado 1X2)
    """
    df = pd.read_csv(url, parse_dates=["date"])

    # --- Filtro temporal: solo partidos desde anio_min ---
    df = df[df["date"].dt.year >= anio_min].copy()

    # --- Limpieza de valores faltantes en el marcador ---
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # --- Normalización de nombres de equipos ---
    df["home_team"] = df["home_team"].str.strip()
    df["away_team"] = df["away_team"].str.strip()

    # --- Campo neutral como booleano seguro ---
    df["neutral"] = df["neutral"].astype(bool)

    # --- Variable auxiliar de resultado (para medir accuracy) ---
    def resultado_1x2(row):
        if row["home_score"] > row["away_score"]:
            return "H"   # gana local
        elif row["home_score"] < row["away_score"]:
            return "A"   # gana visitante
        return "D"       # empate

    df["result"] = df.apply(resultado_1x2, axis=1)
    df = df.sort_values("date").reset_index(drop=True)

    print(f"[datos] Partidos cargados (>= {anio_min}): {len(df)}")
    print(f"[datos] Rango de fechas: {df['date'].min().date()}  ->  {df['date'].max().date()}")
    print(f"[datos] Selecciones distintas: "
          f"{pd.unique(df[['home_team', 'away_team']].values.ravel()).size}")
    return df


# =========================================================================================
# 4. CONFIGURACIÓN DEL PARTIDO
# =========================================================================================
# Cambia aquí el enfrentamiento que quieres predecir.
HOME_TEAM = "Argentina"
AWAY_TEAM = "Brazil"
NEUTRAL   = False     # True si se juega en cancha neutral (sin ventaja de localía)
MAX_GOALS = 8         # tamaño de la matriz de marcadores (0..MAX_GOALS por equipo)

# Proporción del dataset reservada para test (medición de accuracy).
TEST_FRACTION = 0.15


# =========================================================================================
# 5. DIXON-COLES  (para variables de XGBoost)
# =========================================================================================
# El modelo de Dixon-Coles (1997) estima, por máxima verosimilitud, la fuerza
# OFENSIVA (attack) y DEFENSIVA (defence) de cada selección, una ventaja de
# localía global y un parámetro de corrección "rho" para marcadores bajos.
#
#   lambda_local     = exp( intercepto + ataque_local  - defensa_visit + localia*(1-neutral) )
#   lambda_visitante = exp( intercepto + ataque_visit  - defensa_local )
#
# Estas fuerzas (attack/defence) son las VARIABLES que luego alimentan a XGBoost.

def _dc_tau(x, y, lam, mu, rho):
    """Corrección de Dixon-Coles para los marcadores bajos (0-0, 1-0, 0-1, 1-1)."""
    x = np.asarray(x); y = np.asarray(y)
    out = np.ones_like(lam, dtype=float)
    out = np.where((x == 0) & (y == 0), 1.0 - lam * mu * rho, out)
    out = np.where((x == 0) & (y == 1), 1.0 + lam * rho,       out)
    out = np.where((x == 1) & (y == 0), 1.0 + mu * rho,        out)
    out = np.where((x == 1) & (y == 1), 1.0 - rho,             out)
    return out


def ajustar_dixon_coles(df, xi=0.0):
    """Ajusta Dixon-Coles por MLE y devuelve un dict con fuerzas por equipo.

    xi : factor de decaimiento temporal (0 = sin ponderación). Si xi>0 los
         partidos más antiguos pesan menos: w = exp(-xi * dias_atras / 365).
    """
    teams = np.sort(pd.unique(df[["home_team", "away_team"]].values.ravel()))
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}

    hi = df["home_team"].map(idx).to_numpy()
    ai = df["away_team"].map(idx).to_numpy()
    hg = df["home_score"].to_numpy()
    ag = df["away_score"].to_numpy()
    not_neutral = (~df["neutral"]).to_numpy().astype(float)

    # Ponderación temporal
    if xi > 0:
        dias = (df["date"].max() - df["date"]).dt.days.to_numpy()
        w = np.exp(-xi * dias / 365.0)
    else:
        w = np.ones(len(df))

    # Vector de parámetros: [attack(n), defence(n), home_adv, rho]
    # Restricción sum-to-zero sobre attack para identificabilidad.
    def neg_log_like(params):
        attack = params[:n]
        defence = params[n:2 * n]
        home_adv = params[2 * n]
        rho = params[2 * n + 1]

        # centramos el ataque (identificabilidad)
        attack = attack - attack.mean()

        log_lam_h = attack[hi] - defence[ai] + home_adv * not_neutral
        log_lam_a = attack[ai] - defence[hi]
        lam_h = np.exp(log_lam_h)
        lam_a = np.exp(log_lam_a)

        tau = _dc_tau(hg, ag, lam_h, lam_a, rho)
        tau = np.clip(tau, 1e-10, None)

        ll = (np.log(tau)
              + poisson.logpmf(hg, lam_h)
              + poisson.logpmf(ag, lam_a))
        return -np.sum(w * ll)

    x0 = np.concatenate([
        np.zeros(n),          # attack
        np.zeros(n),          # defence
        np.array([0.25]),     # home_adv inicial
        np.array([-0.1]),     # rho inicial
    ])
    bounds = [(-3, 3)] * (2 * n) + [(-2, 2), (-0.2, 0.2)]

    print("[dixon-coles] Ajustando por máxima verosimilitud "
          f"({n} selecciones, {2*n+2} parámetros)...")
    res = minimize(neg_log_like, x0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 200, "disp": False})

    attack = res.x[:n] - res.x[:n].mean()
    defence = res.x[n:2 * n]
    home_adv = res.x[2 * n]
    rho = res.x[2 * n + 1]

    strengths = pd.DataFrame({
        "team": teams,
        "attack": attack,
        "defence": defence,
    }).set_index("team")

    print(f"[dixon-coles] Ventaja de localía (gamma) = {home_adv:.3f} | rho = {rho:.3f}")
    return {
        "strengths": strengths,
        "home_adv": home_adv,
        "rho": rho,
        "teams": teams,
    }


# =========================================================================================
# 6. MODELO 1: MCMC BAYESIANO  (función reutilizable)  -- Baio-Blangiardo
# =========================================================================================
# Modelo Poisson jerárquico (Baio & Blangiardo, 2010) muestreado con PyMC.
#
#   goles_local     ~ Poisson(theta_local)
#   goles_visitante ~ Poisson(theta_visit)
#   log(theta_local) = intercepto + home + att[local]  + def[visit]
#   log(theta_visit) = intercepto        + att[visit]  + def[local]
#
#   att[t], def[t] con priors jerárquicos (media y sd aprendidas de los datos)
#   y restricción sum-to-zero.
#
# Para el marcador usamos la DISTRIBUCIÓN PREDICTIVA POSTERIOR: por cada muestra
# del posterior construimos la matriz de Poisson y promediamos sobre todas las
# muestras -> el marcador refleja la incertidumbre del modelo.

def entrenar_bayesiano(df, draws=1000, tune=1000, chains=2, target_accept=0.9):
    """Entrena el modelo jerárquico bayesiano y devuelve el objeto necesario
    para predecir (idata + índices de equipos)."""
    import pymc as pm  # import diferido

    teams = np.sort(pd.unique(df[["home_team", "away_team"]].values.ravel()))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    hi = df["home_team"].map(idx).to_numpy()
    ai = df["away_team"].map(idx).to_numpy()
    hg = df["home_score"].to_numpy()
    ag = df["away_score"].to_numpy()
    not_neutral = (~df["neutral"]).to_numpy().astype(float)

    print(f"[bayes] Entrenando modelo jerárquico PyMC "
          f"(draws={draws}, tune={tune}, chains={chains})...")

    with pm.Model() as model:
        # Hiper-priors (jerarquía)
        intercept = pm.Normal("intercept", mu=0.0, sigma=1.0)
        home = pm.Normal("home", mu=0.0, sigma=1.0)

        sd_att = pm.HalfNormal("sd_att", sigma=1.0)
        sd_def = pm.HalfNormal("sd_def", sigma=1.0)

        att_star = pm.Normal("att_star", mu=0.0, sigma=sd_att, shape=n)
        def_star = pm.Normal("def_star", mu=0.0, sigma=sd_def, shape=n)

        # Restricción sum-to-zero (identificabilidad)
        att = pm.Deterministic("att", att_star - pm.math.mean(att_star))
        defs = pm.Deterministic("def", def_star - pm.math.mean(def_star))

        theta_h = pm.math.exp(intercept + home * not_neutral + att[hi] + defs[ai])
        theta_a = pm.math.exp(intercept + att[ai] + defs[hi])

        pm.Poisson("home_goals", mu=theta_h, observed=hg)
        pm.Poisson("away_goals", mu=theta_a, observed=ag)

        idata = pm.sample(draws=draws, tune=tune, chains=chains,
                          target_accept=target_accept, random_seed=RANDOM_SEED,
                          progressbar=True)

    return {"idata": idata, "idx": idx, "teams": teams}


def predecir_bayesiano(modelo, home_team, away_team, neutral=False, max_goals=MAX_GOALS):
    """Función reutilizable: matriz de probabilidad de marcador (predictiva posterior).

    Devuelve una matriz (max_goals+1, max_goals+1) donde la celda [i, j] es
    P(local marca i, visitante marca j) promediada sobre el posterior.
    """
    idata = modelo["idata"]
    idx = modelo["idx"]
    if home_team not in idx or away_team not in idx:
        raise ValueError("Alguna selección no está en el dataset de entrenamiento.")

    post = idata.posterior
    # aplanamos cadenas y draws -> (S muestras,)
    intercept = post["intercept"].values.reshape(-1)
    home = post["home"].values.reshape(-1)
    att = post["att"].values.reshape(-1, post["att"].shape[-1])
    defs = post["def"].values.reshape(-1, post["def"].shape[-1])

    h, a = idx[home_team], idx[away_team]
    nn = 0.0 if neutral else 1.0

    lam_h = np.exp(intercept + home * nn + att[:, h] + defs[:, a])  # (S,)
    lam_a = np.exp(intercept + att[:, a] + defs[:, h])              # (S,)

    goals = np.arange(max_goals + 1)
    # pmf por muestra: (S, max_goals+1)
    pmf_h = poisson.pmf(goals[None, :], lam_h[:, None])
    pmf_a = poisson.pmf(goals[None, :], lam_a[:, None])

    # matriz por muestra y promedio sobre el posterior (predictiva posterior)
    # matriz[i,j] = mean_s( pmf_h[s,i] * pmf_a[s,j] )
    matrix = np.einsum("si,sj->ij", pmf_h, pmf_a) / pmf_h.shape[0]
    matrix /= matrix.sum()  # normaliza (corta la cola > max_goals)
    return matrix


# =========================================================================================
# 7. VARIABLES Y REGRESORES DE XGBOOST DE GOLES
# =========================================================================================
# Enfoque híbrido de Groll: XGBoost NO produce un marcador por sí solo.
# Entrenamos DOS regresores con objetivo Poisson (count:poisson) que predicen
# los GOLES ESPERADOS de cada equipo, usando como variables las fuerzas
# ofensiva/defensiva estimadas por Dixon-Coles. Después convertimos esas tasas
# de gol en una matriz de marcadores con la distribución de Poisson.

def construir_variables_xgb(df, dc):
    """Construye la matriz de variables (X) y los objetivos (y) para los dos
    regresores, a partir de las fuerzas de Dixon-Coles.

    Variables por partido:
      - att_home, def_home : fuerza ofensiva/defensiva del local
      - att_away, def_away : fuerza ofensiva/defensiva del visitante
      - home_flag          : 1 si hay localía (no neutral), 0 si es neutral
    Objetivos:
      - y_home : goles del local
      - y_away : goles del visitante
    """
    s = dc["strengths"]
    # equipos vistos por Dixon-Coles
    mask = df["home_team"].isin(s.index) & df["away_team"].isin(s.index)
    d = df[mask].copy()

    att_home = s.loc[d["home_team"], "attack"].to_numpy()
    def_home = s.loc[d["home_team"], "defence"].to_numpy()
    att_away = s.loc[d["away_team"], "attack"].to_numpy()
    def_away = s.loc[d["away_team"], "defence"].to_numpy()
    home_flag = (~d["neutral"]).astype(int).to_numpy()

    X = pd.DataFrame({
        "att_home": att_home,
        "def_home": def_home,
        "att_away": att_away,
        "def_away": def_away,
        "home_flag": home_flag,
    }, index=d.index)

    y_home = d["home_score"].to_numpy()
    y_away = d["away_score"].to_numpy()
    return X, y_home, y_away


def _vector_variables(dc, home_team, away_team, neutral):
    """Vector de variables (1 fila) para un enfrentamiento concreto."""
    s = dc["strengths"]
    return pd.DataFrame([{
        "att_home": s.loc[home_team, "attack"],
        "def_home": s.loc[home_team, "defence"],
        "att_away": s.loc[away_team, "attack"],
        "def_away": s.loc[away_team, "defence"],
        "home_flag": 0 if neutral else 1,
    }])


def entrenar_regresores_xgb(X, y_home, y_away):
    """Entrena los DOS regresores Poisson de XGBoost (goles local y visitante)."""
    params = dict(
        objective="count:poisson",   # regresión de conteos (goles)
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=3,
        random_state=RANDOM_SEED,
    )
    print("[xgboost] Entrenando regresor de goles del LOCAL...")
    reg_home = xgb.XGBRegressor(**params).fit(X, y_home)
    print("[xgboost] Entrenando regresor de goles del VISITANTE...")
    reg_away = xgb.XGBRegressor(**params).fit(X, y_away)
    return {"reg_home": reg_home, "reg_away": reg_away}


# =========================================================================================
# 8. ENTRENAR REGRESORES DE XGBOOST
# =========================================================================================
# (Se ejecuta en la sección 11; aquí queda la función reutilizable del modelo 2.)


# =========================================================================================
# 9. MODELO 2: XGBOOST  (función reutilizable)
# =========================================================================================
def predecir_xgboost(modelos_xgb, dc, home_team, away_team,
                     neutral=False, max_goals=MAX_GOALS):
    """Función reutilizable: matriz de probabilidad de marcador con XGBoost híbrido.

    1) XGBoost estima goles esperados (lambda) de cada equipo.
    2) La estructura de Poisson convierte esas tasas en la matriz de marcadores.
    """
    x = _vector_variables(dc, home_team, away_team, neutral)
    lam_h = float(modelos_xgb["reg_home"].predict(x)[0])
    lam_a = float(modelos_xgb["reg_away"].predict(x)[0])
    lam_h = max(lam_h, 1e-6)
    lam_a = max(lam_a, 1e-6)

    goals = np.arange(max_goals + 1)
    pmf_h = poisson.pmf(goals, lam_h)
    pmf_a = poisson.pmf(goals, lam_a)
    matrix = np.outer(pmf_h, pmf_a)
    matrix /= matrix.sum()
    return matrix, lam_h, lam_a


# =========================================================================================
# 10. FUNCIÓN DE VISUALIZACIÓN
# =========================================================================================
def _resumen_matriz(matrix):
    """Devuelve P(local), P(empate), P(visitante) y el marcador más probable."""
    p_home = np.tril(matrix, -1).sum()   # local > visitante
    p_draw = np.trace(matrix)            # iguales
    p_away = np.triu(matrix, 1).sum()    # visitante > local
    i, j = np.unravel_index(np.argmax(matrix), matrix.shape)
    return p_home, p_draw, p_away, (i, j, matrix[i, j])


def visualizar(matrix, home_team, away_team, titulo, lam=None):
    """Dibuja la matriz de probabilidad de marcadores como heatmap + resumen 1X2."""
    p_home, p_draw, p_away, (bi, bj, bp) = _resumen_matriz(matrix)

    cmap = LinearSegmentedColormap.from_list("kiro", ["#ffffff", "#2c7fb8", "#08306b"])
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(matrix, cmap=cmap, origin="upper")

    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_xlabel(f"Goles {away_team} (visitante)")
    ax.set_ylabel(f"Goles {home_team} (local)")

    # anota cada celda con su probabilidad (%)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if val >= 0.005:
                ax.text(j, i, f"{val*100:.0f}", ha="center", va="center",
                        color="white" if val > matrix.max() * 0.5 else "black",
                        fontsize=8)
    # resalta el marcador más probable
    ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False,
                               edgecolor="#e6550d", lw=2.5))

    sub = (f"1 ({home_team}): {p_home*100:.1f}%   "
           f"X (empate): {p_draw*100:.1f}%   "
           f"2 ({away_team}): {p_away*100:.1f}%\n"
           f"Marcador más probable: {bi}-{bj}  ({bp*100:.1f}%)")
    if lam is not None:
        sub += f"\nGoles esperados: {home_team} {lam[0]:.2f} - {lam[1]:.2f} {away_team}"

    ax.set_title(f"{titulo}\n{home_team} vs {away_team}\n{sub}", fontsize=11)
    fig.colorbar(im, ax=ax, label="Probabilidad")
    plt.tight_layout()
    return fig


# ----- Medición de accuracy -----------------------------------------------------------
def _outcome_de_matriz(matrix):
    p_home = np.tril(matrix, -1).sum()
    p_draw = np.trace(matrix)
    p_away = np.triu(matrix, 1).sum()
    return ["H", "D", "A"][int(np.argmax([p_home, p_draw, p_away]))]


def accuracy_bayes(modelo, df_test, max_goals=MAX_GOALS):
    idx = modelo["idx"]
    ok = total = 0
    for _, r in df_test.iterrows():
        if r["home_team"] not in idx or r["away_team"] not in idx:
            continue
        m = predecir_bayesiano(modelo, r["home_team"], r["away_team"],
                               bool(r["neutral"]), max_goals)
        ok += (_outcome_de_matriz(m) == r["result"]); total += 1
    return ok / total if total else float("nan"), total


def accuracy_xgb(modelos_xgb, dc, df_test, max_goals=MAX_GOALS):
    s = dc["strengths"]
    ok = total = 0
    for _, r in df_test.iterrows():
        if r["home_team"] not in s.index or r["away_team"] not in s.index:
            continue
        m, _, _ = predecir_xgboost(modelos_xgb, dc, r["home_team"], r["away_team"],
                                   bool(r["neutral"]), max_goals)
        ok += (_outcome_de_matriz(m) == r["result"]); total += 1
    return ok / total if total else float("nan"), total


# =========================================================================================
# 11. EJECUTAR MODELOS Y VISUALIZAR RESULTADOS
# =========================================================================================
def main():
    # --- 3. Datos ---
    df = cargar_datos(anio_min=2018)

    # --- Split temporal train / test (para accuracy) ---
    corte = int(len(df) * (1 - TEST_FRACTION))
    df_train = df.iloc[:corte].copy()
    df_test = df.iloc[corte:].copy()
    print(f"[split] train={len(df_train)}  test={len(df_test)} "
          f"(desde {df_test['date'].min().date()})")

    # Validación de la configuración del partido
    equipos = set(pd.unique(df_train[["home_team", "away_team"]].values.ravel()))
    if HOME_TEAM not in equipos or AWAY_TEAM not in equipos:
        raise SystemExit(f"Configura HOME_TEAM/AWAY_TEAM con selecciones válidas. "
                         f"Ej.: {sorted(list(equipos))[:10]} ...")

    # --- 5. Dixon-Coles (variables para XGBoost) ---
    dc = ajustar_dixon_coles(df_train, xi=0.0)
    s = dc["strengths"]
    print("\n[dixon-coles] Fuerzas de las selecciones configuradas:")
    print(s.loc[[HOME_TEAM, AWAY_TEAM]].round(3))

    # --- 7-8. Variables y entrenamiento de los regresores XGBoost ---
    X, y_home, y_away = construir_variables_xgb(df_train, dc)
    modelos_xgb = entrenar_regresores_xgb(X, y_home, y_away)

    # --- 6. Modelo 1: MCMC Bayesiano ---
    modelo_bayes = entrenar_bayesiano(df_train, draws=1000, tune=1000, chains=2)

    # ====================== Predicciones del partido configurado ======================
    print("\n" + "=" * 70)
    print(f"PREDICCIÓN: {HOME_TEAM} vs {AWAY_TEAM} (neutral={NEUTRAL})")
    print("=" * 70)

    # Modelo 1
    M_bayes = predecir_bayesiano(modelo_bayes, HOME_TEAM, AWAY_TEAM, NEUTRAL, MAX_GOALS)
    # Modelo 2
    M_xgb, lam_h, lam_a = predecir_xgboost(modelos_xgb, dc, HOME_TEAM, AWAY_TEAM,
                                           NEUTRAL, MAX_GOALS)

    for nombre, M in [("MCMC Bayesiano", M_bayes), ("XGBoost híbrido", M_xgb)]:
        ph, pd_, pa, (bi, bj, bp) = _resumen_matriz(M)
        print(f"\n[{nombre}]")
        print(f"  1X2 -> {HOME_TEAM}: {ph*100:5.1f}% | Empate: {pd_*100:5.1f}% | "
              f"{AWAY_TEAM}: {pa*100:5.1f}%")
        print(f"  Marcador más probable: {bi}-{bj} ({bp*100:.1f}%)")

    # --- 10. Visualización ---
    fig1 = visualizar(M_bayes, HOME_TEAM, AWAY_TEAM,
                      "Modelo 1 — MCMC Bayesiano (Baio-Blangiardo)")
    fig2 = visualizar(M_xgb, HOME_TEAM, AWAY_TEAM,
                      "Modelo 2 — XGBoost híbrido (Groll)", lam=(lam_h, lam_a))
    fig1.savefig("matriz_bayes.png", dpi=120)
    fig2.savefig("matriz_xgboost.png", dpi=120)
    print("\n[plot] Guardado: matriz_bayes.png, matriz_xgboost.png")
    plt.show()

    # ============================ Accuracy sobre el test ============================
    print("\n" + "=" * 70)
    print("ACCURACY (1X2) SOBRE EL CONJUNTO DE TEST")
    print("=" * 70)
    acc_b, n_b = accuracy_bayes(modelo_bayes, df_test, MAX_GOALS)
    acc_x, n_x = accuracy_xgb(modelos_xgb, dc, df_test, MAX_GOALS)

    # Baseline: predecir siempre local
    base = (df_test["result"] == "H").mean()

    print(f"  Baseline (siempre local):  {base*100:5.2f}%")
    print(f"  MCMC Bayesiano:            {acc_b*100:5.2f}%   (n={n_b})")
    print(f"  XGBoost híbrido:           {acc_x*100:5.2f}%   (n={n_x})")


if __name__ == "__main__":
    main()
