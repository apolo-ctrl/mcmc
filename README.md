# Predicción de marcadores de selecciones nacionales

Pipeline en Python para predecir el resultado de partidos entre selecciones usando el
dataset [`martj42/international_results`](https://github.com/martj42/international_results).
Combina **Dixon-Coles** para estimar la fuerza de cada selección con **dos modelos de
machine learning** que predicen la matriz de probabilidad de cada marcador posible:

1. **MCMC Bayesiano (Baio-Blangiardo)** — modelo Poisson jerárquico muestreado con PyMC.
   El marcador se obtiene de la **distribución predictiva posterior** (se promedia la
   matriz de marcadores sobre todas las muestras del posterior, reflejando la incertidumbre
   del modelo).
2. **XGBoost híbrido (Groll)** — XGBoost no produce un marcador por sí solo, así que se
   entrenan **dos regresores con objetivo Poisson** (`count:poisson`) que predicen los
   **goles esperados** de cada equipo; esas tasas se convierten en una matriz de marcadores
   con la **distribución de Poisson**.

La salida es una **matriz de probabilidad de cada marcador**, las probabilidades **1X2**, el
marcador más probable y el **accuracy** de ambos modelos sobre un conjunto de test.

---

## Dataset

- Repositorio: https://github.com/martj42/international_results
- CSV directo: `https://raw.githubusercontent.com/martj42/international_results/master/results.csv`

Se limpian los datos, se filtran los partidos **desde 2018** y se calculan variables
intermedias con Dixon-Coles.

---

## Instalación

```bash
pip install -r requirements.txt
```

Requiere Python 3.10+. Las dependencias principales son `pandas`, `numpy`, `scipy`,
`xgboost`, `scikit-learn`, `pymc`, `arviz` y `matplotlib`.

> Nota: la API de scikit-learn de XGBoost (`XGBRegressor`) **requiere** que
> `scikit-learn` esté instalado. Si ves `ImportError: sklearn needs to be installed`,
> ejecuta `pip install scikit-learn`.

---

## Uso

```bash
python src/prediccion_selecciones.py
```

Para elegir el enfrentamiento, edita la sección **4. Configuración del partido** en
`src/prediccion_selecciones.py`:

```python
HOME_TEAM = "Argentina"
AWAY_TEAM = "Brazil"
NEUTRAL   = False     # True si se juega en cancha neutral (sin ventaja de localía)
MAX_GOALS = 8         # tamaño de la matriz de marcadores
```

### Google Colab

1. `!pip install -r requirements.txt` (o instala los paquetes manualmente).
2. Sube `src/prediccion_selecciones.py` y ejecuta `!python prediccion_selecciones.py`,
   o pega el contenido por bloques (cada sección está marcada con `# === N. ... ===`).

---

## Estructura del pipeline

El script sigue estas secciones, en orden:

| #  | Sección                                   | Qué hace |
|----|-------------------------------------------|----------|
| 1  | Instalar librerías                        | Comando `pip install` (comentado para Colab) |
| 2  | Importar librerías                        | Imports; PyMC se importa de forma diferida |
| 3  | Descargar el dataset                      | Descarga, limpia y filtra (`>= 2018`); split train/test temporal |
| 4  | Configuración del partido                 | `HOME_TEAM`, `AWAY_TEAM`, `NEUTRAL`, `MAX_GOALS` |
| 5  | Dixon-Coles                               | MLE de fuerza ofensiva/defensiva + localía + `rho` (variables para XGBoost) |
| 6  | Modelo 1: MCMC Bayesiano                  | Poisson jerárquico (PyMC) + predictiva posterior |
| 7  | Variables y regresores de XGBoost         | Construye `X`/`y` a partir de las fuerzas de Dixon-Coles |
| 8  | Entrenar regresores de XGBoost            | Dos regresores `count:poisson` (goles local y visitante) |
| 9  | Modelo 2: XGBoost                         | Tasas de gol -> matriz de marcadores con Poisson |
| 10 | Función de visualización                  | Heatmap de la matriz + resumen 1X2 |
| 11 | Ejecutar y visualizar                     | Predicción del partido + accuracy en test |

---

## Detalles de modelado

### Dixon-Coles (sección 5)
Estima por máxima verosimilitud la fuerza **ofensiva** (`attack`) y **defensiva**
(`defence`) de cada selección, una **ventaja de localía** global `gamma` (aplicada solo si
el partido no es neutral) y el parámetro `rho` que corrige los marcadores bajos
(0-0, 1-0, 0-1, 1-1). Soporta **ponderación temporal** opcional (`xi`) para dar más peso a
los partidos recientes.

### Modelo 1 — MCMC Bayesiano (sección 6)
Modelo de Baio-Blangiardo:

```
goles_local     ~ Poisson(theta_local)
goles_visitante ~ Poisson(theta_visit)
log(theta_local) = intercepto + home + att[local] + def[visit]
log(theta_visit) = intercepto        + att[visit] + def[local]
```

`att[t]` y `def[t]` tienen priors jerárquicos con restricción *sum-to-zero*. El marcador se
obtiene promediando la matriz de Poisson sobre todas las muestras del posterior.

### Modelo 2 — XGBoost híbrido (secciones 7-9)
Enfoque de Groll: dos regresores `XGBRegressor(objective="count:poisson")` predicen los
goles esperados de cada equipo usando como variables las fuerzas de Dixon-Coles
(`att_home`, `def_home`, `att_away`, `def_away`, `home_flag`). La estructura de Poisson
convierte esas tasas en la matriz de marcadores.

### Accuracy (sección 11)
Para cada partido del conjunto de test se deriva el resultado 1X2 más probable de la matriz
de cada modelo y se compara con el resultado real, contrastándolo con un baseline
("siempre gana el local").

---

## Notas prácticas

- **PyMC** es la parte más lenta. Empieza con `draws=1000, tune=1000, chains=2`; baja a
  `draws=500` si tarda demasiado.
- El ajuste de Dixon-Coles optimiza `2*N+2` parámetros (N = nº de selecciones). Activa la
  ponderación temporal con `ajustar_dixon_coles(df_train, xi=0.3)` si quieres priorizar
  partidos recientes.
- En accuracy 1X2 con selecciones es habitual moverse en torno al ~50% (los empates son
  difíciles de predecir); lo relevante es superar al baseline.

---

## Liga MX (clubes)

El mismo modelo aplicado a la **Liga MX** está en [`src/liga_mx.py`](src/liga_mx.py). Reutiliza
todas las funciones de `prediccion_selecciones.py` y solo cambia la **fuente de datos**:

- Fuente: [football-data.co.uk — México](https://www.football-data.co.uk/new/MEX.csv)
  (columnas `Date, Home, Away, HG, AG, Res` + cuotas).
- El loader descarga el CSV con un *User-Agent* de navegador (el sitio bloquea las
  peticiones automáticas sin cabecera).

Ejecución:

```bash
python src/liga_mx.py
```

Configura el partido editando `HOME_TEAM`, `AWAY_TEAM` al inicio del archivo. Usa los
**nombres exactos** del dataset; si te equivocas, el script imprime la lista de equipos
disponibles. Fuente alternativa en formato CSV limpio:
[footballcsv/mexico](https://github.com/footballcsv/mexico).

## Actualización automática

El script **siempre usa los datos más recientes**: cada ejecución descarga el CSV en vivo
desde `raw.githubusercontent.com/.../results.csv`, así que no hay copia local cacheada. El
dataset de origen se actualiza cada noche, por lo que basta con **ejecutar el script de
nuevo** para tener los datos al día.

Para ejecutarlo de forma desatendida cada noche hay dos opciones:

### GitHub Actions (en la nube, recomendado)
El repo incluye [`.github/workflows/prediccion.yml`](.github/workflows/prediccion.yml), que
corre cada noche (`cron: "0 7 * * *"`, en UTC), instala dependencias, ejecuta la predicción
y sube los resultados (`prediccion_salida.txt`, `matriz_bayes.png`, `matriz_xgboost.png`)
como *artifacts* descargables desde la pestaña **Actions**. También se puede lanzar a mano
con **Run workflow** (`workflow_dispatch`). Ajusta la hora del `cron` a tu gusto.

### Windows (en tu PC, con el Programador de tareas)
Crea un `.bat` (p. ej. `run_prediccion.bat`):

```bat
cd /d C:\Users\apolo\Downloads\mundial2026\mcmc
git pull
python src/prediccion_selecciones.py > prediccion_salida.txt 2>&1
```

Luego en **Programador de tareas** (Task Scheduler): *Crear tarea básica* → desencadenador
*Diariamente* a la hora deseada → acción *Iniciar un programa* → selecciona el `.bat`.
(Requiere que el PC esté encendido a esa hora.)

## Licencia

MIT. Ver [LICENSE](LICENSE).
