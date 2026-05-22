"""
Modelo Poisson para analisis de partidos de futbol
Con Dixon-Coles correction + xG desde FBref (todo gratis)

Fuente de datos: football-data.co.uk, FBref
"""

import sys
import io
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
except Exception:
    pass

import pandas as pd
import numpy as np
from math import exp, factorial, log
import re
import os
import json
from datetime import datetime

# ──────────────────────────────────────────────
# ALIAS DE EQUIPOS
# ──────────────────────────────────────────────

TEAM_ALIASES = {
    'manchester city': 'Man City', 'manchester united': 'Man United',
    'man utd': 'Man United', 'newcastle united': 'Newcastle',
    'tottenham hotspur': 'Tottenham', 'spurs': 'Tottenham',
    'nottingham forest': "Nott'm Forest", 'notts forest': "Nott'm Forest",
    'bayern': 'Bayern Munich', 'bayern munchen': 'Bayern Munich',
    'eintracht frankfurt': 'Ein Frankfurt', 'borussia dortmund': 'Dortmund',
    'borussia mgladbach': "M'gladbach", 'gladbach': "M'gladbach",
    'rb leipzig': 'RB Leipzig', 'psg': 'Paris SG',
    'paris saint germain': 'Paris SG', 'inter milan': 'Inter',
    'ac milan': 'Milan', 'juventus turin': 'Juventus',
    'real betis': 'Betis', 'athletic bilbao': 'Ath Bilbao',
    'bilbao': 'Ath Bilbao', 'athletic madrid': 'Ath Madrid',
    'atletico madrid': 'Ath Madrid', 'atletico': 'Ath Madrid',
    'real sociedad': 'Sociedad', 'rayo vallecano': 'Vallecano',
    'espanyol': 'Espanol', 'rcd espanyol': 'Espanol',
    'saint etienne': 'St Etienne', 'st etienne': 'St Etienne',
    'le havre': 'Le Havre', 'strasbourg': 'Strasbourg',
}


def resolver_equipo(nombre):
    n = nombre.lower().strip()
    return TEAM_ALIASES.get(n, nombre)


# ──────────────────────────────────────────────
# 1. DATOS DESDE FOOTBALL-DATA.CO.UK
# ──────────────────────────────────────────────

def descargar_liga(liga, temporada):
    url = f"https://www.football-data.co.uk/mmz4281/{temporada}/{liga}.csv"
    try:
        df = pd.read_csv(url).copy()
        df.insert(0, 'Liga', liga)
        print(f"  OK {liga} {temporada}/{temporada+1}: {len(df)} partidos")
        return df
    except Exception as e:
        print(f"  XX {liga} {temporada}/{temporada+1}: {e}")
        return None


def descargar_varias_ligas(ligas, temporada):
    dfs = [descargar_liga(liga, temporada) for liga in ligas]
    dfs = [d for d in dfs if d is not None]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# ──────────────────────────────────────────────
# 2. xG DESDE CSV (GRATIS, MANUAL)
# ──────────────────────────────────────────────
#
# El xG se carga MANUALMENTE desde un archivo CSV.
# Por que? FBref tiene Cloudflare, Understat cambio su API.
# La forma MAS facil de obtener xG gratis:
#
#   1. Ir a https://fbref.com/en/comps/9/Premier-League-Stats
#      (cambiar comps/9 por comps/12 LaLiga, /11 Serie A, /20 Bundesliga, /13 Ligue 1)
#   2. En la tabla "Standard Stats", la 3ra columna es xG y la 4ta xGA
#   3. Copiar nombres de equipos + xG + xGA a un CSV:
#
#       equipo,xg_favor,xg_contra
#       Man City,2.3,0.9
#       Arsenal,1.8,0.8
#       ...
#
#   4. Correr: python poisson_model.py --xg-csv mi_xg.csv "Barcelona vs Real Madrid"
#
# Consejo: FBref tiene xG seasonales completos (no por partido), perfectos para el modelo.


def cargar_xg_csv(ruta_csv):
    """
    Carga xG desde un archivo CSV manual.
    Formato esperado: equipo,xg_favor,xg_contra
    """
    try:
        df = pd.read_csv(ruta_csv)
        xg_data = {}
        for _, row in df.iterrows():
            equipo = str(row.iloc[0]).strip()
            xg_favor = float(row.iloc[1])
            xg_contra = float(row.iloc[2]) if len(row) > 2 else xg_favor * 0.8
            xg_data[equipo] = {'xg_favor': xg_favor, 'xg_contra': xg_contra}
        print(f">> xG cargado desde {ruta_csv}: {len(xg_data)} equipos")
        return xg_data
    except Exception as e:
        print(f"XX Error cargando xG desde CSV: {e}")
        return None


# ──────────────────────────────────────────────
# 3. DIXON-COLES ADJUSTMENT
# ──────────────────────────────────────────────

def dc_tau(x, y, lam, mu, rho):
    """Factor de ajuste Dixon-Coles para resultados de 0-1 goles."""
    if x == 0 and y == 0:
        return 1 - lam * mu * rho
    if x == 0 and y == 1:
        return 1 + lam * rho
    if x == 1 and y == 0:
        return 1 + mu * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def estimar_rho(df_partidos):
    """
    Estima rho (parametro Dixon-Coles) por liga desde datos historicos.
    Usa MLE simple: busca rho que maximiza log-likelihood.
    """
    from scipy.optimize import minimize_scalar

    rho_por_liga = {}
    for liga in df_partidos['Liga'].unique():
        df_liga = df_partidos[df_partidos['Liga'] == liga].copy()
        # Seleccionar solo las columnas que necesitamos
        cols_needed = {'Liga', 'Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'}
        cols_present = [c for c in df_liga.columns if c in cols_needed]
        df_liga = df_liga[cols_present]

        avg_h = df_liga['FTHG'].mean()
        avg_a = df_liga['FTAG'].mean()

        # Attack/defense strengths simplificado
        ataque_h = df_liga.groupby('HomeTeam')['FTHG'].mean() / avg_h if avg_h > 0 else 1
        defensa_h = df_liga.groupby('HomeTeam')['FTAG'].mean() / avg_a if avg_a > 0 else 1
        ataque_a = df_liga.groupby('AwayTeam')['FTAG'].mean() / avg_a if avg_a > 0 else 1
        defensa_a = df_liga.groupby('AwayTeam')['FTHG'].mean() / avg_h if avg_h > 0 else 1

        def neg_log_likelihood(rho):
            ll = 0
            for _, partido in df_liga.iterrows():
                eq_h = partido['HomeTeam']
                eq_a = partido['AwayTeam']
                gh = int(partido['FTHG'])
                ga = int(partido['FTAG'])

                ah = ataque_h.get(eq_h, 1) if hasattr(ataque_h, 'get') else 1
                dh = defensa_h.get(eq_h, 1) if hasattr(defensa_h, 'get') else 1
                aa = ataque_a.get(eq_a, 1) if hasattr(ataque_a, 'get') else 1
                da = defensa_a.get(eq_a, 1) if hasattr(defensa_a, 'get') else 1

                lam = ah * da * avg_h
                mu = aa * dh * avg_a

                p_indep = (lam ** gh * exp(-lam) / factorial(gh)) * \
                          (mu ** ga * exp(-mu) / factorial(ga))
                tau = dc_tau(gh, ga, lam, mu, rho)
                p = p_indep * tau

                if p > 1e-10:
                    ll += log(p)
            return -ll

        try:
            result = minimize_scalar(neg_log_likelihood, bounds=(-0.5, 0.3), method='bounded')
            rho_por_liga[liga] = round(result.x, 4)
            print(f"  DC rho para {liga}: {rho_por_liga[liga]:.4f}")
        except Exception:
            rho_por_liga[liga] = -0.1  # Default si falla
            print(f"  DC rho para {liga}: -0.1000 (default)")

    return rho_por_liga


# ──────────────────────────────────────────────
# 4. CALCULO DE FUERZAS
# ──────────────────────────────────────────────

def calcular_fuerzas(df, xg_data=None, ponderar_forma=True):
    """
    Calcula ataque/defensa local y visitante.
    Si xg_data esta presente, usa xG en vez de goles reales.
    Si ponderar_forma=True, da mas peso a partidos recientes.
    """
    cols_needed = ['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'Liga']
    df = df[[c for c in cols_needed if c in df.columns]].copy()
    df.columns = ['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'Liga']

    usar_xg = xg_data is not None

    if usar_xg:
        # Reemplazar goles reales por xG promedio por equipo
        print(">> Usando xG de FBref en lugar de goles reales")
        for col_local, col_visit, xg_col in [('FTHG', 'FTAG', 'xg_favor')]:
            pass  # Abajo lo hacemos por equipo
        # Asignar xG a cada partido basado en el equipo
        def get_xg(equipo, col):
            if equipo in xg_data:
                return xg_data[equipo][col]
            return None

        df['FTHG_xg'] = df['HomeTeam'].map(lambda t: get_xg(t, 'xg_favor'))
        df['FTAG_xg'] = df['AwayTeam'].map(lambda t: get_xg(t, 'xg_contra'))

        # Donde no hay xG, usar goles reales
        df['FTHG'] = df['FTHG_xg'].fillna(df['FTHG'])
        df['FTAG'] = df['FTAG_xg'].fillna(df['FTAG'])
        df = df.drop(columns=['FTHG_xg', 'FTAG_xg'])

    # Promedios de liga
    promedios = df.groupby('Liga').agg(
        home_goles_avg=('FTHG', 'mean'),
        away_goles_avg=('FTAG', 'mean')
    ).to_dict('index')

    # Ponderacion por forma reciente
    if ponderar_forma and 'Date' in df.columns:
        try:
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            fecha_max = df['Date'].max()
            if pd.notna(fecha_max):
                df['dias_atras'] = (fecha_max - df['Date']).dt.days
                df['peso'] = np.maximum(0, 1 - df['dias_atras'] / 180)  # 6 meses
                df['peso'] = 0.3 + 0.7 * df['peso']  # Minimo 30% de peso
            else:
                df['peso'] = 1.0
        except Exception:
            df['peso'] = 1.0
    else:
        df['peso'] = 1.0

    equipos = []

    for liga in df['Liga'].unique():
        df_liga = df[df['Liga'] == liga]
        avg_home = promedios[liga]['home_goles_avg']
        avg_away = promedios[liga]['away_goles_avg']

        # Local - ponderado
        local = df_liga.groupby('HomeTeam').apply(
            lambda g: pd.Series({
                'goles_favor': (g['FTHG'] * g['peso']).sum(),
                'goles_contra': (g['FTAG'] * g['peso']).sum(),
                'partidos': g['peso'].sum(),
            })
        ).reset_index()

        for _, row in local.iterrows():
            if row['partidos'] < 3:
                continue
            equipos.append({
                'Equipo': row['HomeTeam'],
                'Liga': liga,
                'Attack_Strength_Home': row['goles_favor'] / row['partidos'] / avg_home if avg_home > 0 else 1,
                'Defense_Strength_Home': row['goles_contra'] / row['partidos'] / avg_away if avg_away > 0 else 1,
            })

        # Visitante - ponderado
        visitante = df_liga.groupby('AwayTeam').apply(
            lambda g: pd.Series({
                'goles_favor': (g['FTAG'] * g['peso']).sum(),
                'goles_contra': (g['FTHG'] * g['peso']).sum(),
                'partidos': g['peso'].sum(),
            })
        ).reset_index()

        for _, row in visitante.iterrows():
            if row['partidos'] < 3:
                continue
            existing = [e for e in equipos if e['Equipo'] == row['AwayTeam'] and e['Liga'] == liga]
            if existing:
                existing[0]['Attack_Strength_Away'] = row['goles_favor'] / row['partidos'] / avg_away if avg_away > 0 else 1
                existing[0]['Defense_Strength_Away'] = row['goles_contra'] / row['partidos'] / avg_home if avg_home > 0 else 1
            else:
                equipos.append({
                    'Equipo': row['AwayTeam'],
                    'Liga': liga,
                    'Attack_Strength_Away': row['goles_favor'] / row['partidos'] / avg_away if avg_away > 0 else 1,
                    'Defense_Strength_Away': row['goles_contra'] / row['partidos'] / avg_home if avg_home > 0 else 1,
                    'Attack_Strength_Home': 1,
                    'Defense_Strength_Home': 1,
                })

    return pd.DataFrame(equipos), promedios


# ──────────────────────────────────────────────
# 5. POISSON + DIXON-COLES
# ──────────────────────────────────────────────

def poisson_prob(lam, k):
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return (lam ** k) * exp(-lam) / factorial(k)


def calcular_probabilidades(xg_local, xg_visitante, rho=None, max_goles=10):
    """
    Calcula matriz de probabilidades con Dixon-Coles opcional.
    Si rho es None, usa Poisson independiente (sin ajuste).
    """
    prob_local = 0.0
    prob_empate = 0.0
    prob_visitante = 0.0
    over_25 = 0.0
    scorelines = {}

    for g_local in range(max_goles + 1):
        for g_visit in range(max_goles + 1):
            p_home = poisson_prob(xg_local, g_local)
            p_away = poisson_prob(xg_visitante, g_visit)
            p = p_home * p_away

            if rho is not None:
                p *= dc_tau(g_local, g_visit, xg_local, xg_visitante, rho)

            if p < 0.0001:
                continue

            scorelines[f"{g_local}-{g_visit}"] = p

            if g_local > g_visit:
                prob_local += p
            elif g_local == g_visit:
                prob_empate += p
            else:
                prob_visitante += p

            if g_local + g_visit > 2.5:
                over_25 += p

    return prob_local, prob_empate, prob_visitante, over_25, scorelines


# ──────────────────────────────────────────────
# 6. PREDECIR PARTIDO
# ──────────────────────────────────────────────

def predecir_partido(local, visitante, fuerzas, promedios, liga, rho=None):
    if liga not in promedios:
        return None

    avg_home = promedios[liga]['home_goles_avg']
    avg_away = promedios[liga]['away_goles_avg']

    f_local = fuerzas[(fuerzas['Equipo'] == local) & (fuerzas['Liga'] == liga)]
    f_visit = fuerzas[(fuerzas['Equipo'] == visitante) & (fuerzas['Liga'] == liga)]

    if f_local.empty or f_visit.empty:
        return None

    f_local = f_local.iloc[0]
    f_visit = f_visit.iloc[0]

    xg_local = f_local['Attack_Strength_Home'] * f_visit['Defense_Strength_Away'] * avg_home
    xg_visitante = f_visit['Attack_Strength_Away'] * f_local['Defense_Strength_Home'] * avg_away

    prob_local, prob_empate, prob_visit, over_25, scorelines = calcular_probabilidades(
        xg_local, xg_visitante, rho=rho
    )

    nombre_dc = "Dixon-Coles" if rho is not None else "Poisson"
    ajuste = f" ({nombre_dc}, rho={rho:.3f})" if rho is not None else ""

    return {
        'local': local,
        'visitante': visitante,
        'liga': liga,
        'modelo': f"Poisson{ajuste}",
        'xg_local': round(xg_local, 3),
        'xg_visitante': round(xg_visitante, 3),
        'prob_local': round(prob_local, 4),
        'prob_empate': round(prob_empate, 4),
        'prob_visitante': round(prob_visit, 4),
        'over_2_5': round(over_25, 4),
        'cuota_justa_local': round(1 / prob_local, 2) if prob_local > 0 else 999,
        'cuota_justa_empate': round(1 / prob_empate, 2) if prob_empate > 0 else 999,
        'cuota_justa_visitante': round(1 / prob_visit, 2) if prob_visit > 0 else 999,
        'scorelines': dict(sorted(scorelines.items(), key=lambda x: x[1], reverse=True)[:5]),
        'rho': rho,
    }


# ──────────────────────────────────────────────
# 7. VALUE BET + KELLY
# ──────────────────────────────────────────────

def encontrar_value_bets(pred, cuota_local, cuota_empate, cuota_visitante):
    results = []
    for label, prob, cuota in [
        (f"{pred['local']} (local)", pred['prob_local'], cuota_local),
        ('Empate', pred['prob_empate'], cuota_empate),
        (f"{pred['visitante']} (visitante)", pred['prob_visitante'], cuota_visitante),
    ]:
        if prob > 0:
            ev = (prob * cuota) - 1
            results.append({
                'mercado': label,
                'cuota_casa': cuota,
                'cuota_justa': round(1 / prob, 2),
                'prob_modelo': prob,
                'ev': round(ev, 4),
                'es_value': ev > 0.05,
            })
    return results


def kelly_criterion(prob, cuota, bankroll=1000, fraccion=0.25):
    stake_pct = (prob * cuota - 1) / (cuota - 1) if cuota > 1 else 0
    stake_pct = max(0, stake_pct * fraccion)
    return round(bankroll * stake_pct, 2)


# ──────────────────────────────────────────────
# 8. CARGA COMPLETA DE DATOS
# ──────────────────────────────────────────────

LIGAS = {
    'E0': 'Premier League', 'SP1': 'LaLiga', 'I1': 'Serie A',
    'D1': 'Bundesliga', 'F1': 'Ligue 1',
}


def cargar_datos(force_download=False, usar_xg=False, xg_csv=None):
    """Carga datos de football-data.co.uk + opcional xG + Dixon-Coles."""
    cache_file = 'datos_cache.parquet'
    fuerzas_cache = 'fuerzas_cache.parquet'
    promedios_cache = 'promedios_cache.json'
    rho_cache = 'rho_cache.json'

    # Cargar xG desde CSV
    if xg_csv:
        xg_data = cargar_xg_csv(xg_csv)
        if not xg_data:
            print("X No se pudo cargar xG desde CSV. Usando goles reales.")
            usar_xg = False
        else:
            usar_xg = True

    usar_cache = not force_download

    # Cargar xG solo si ya tenemos datos (desde CSV arriba)
    xg_data = None
    if usar_xg and not xg_data:
        print("  ! --xg requiere --xg-csv <archivo> (no hay scraping automatico disponible)")
        print("  ! FBref y Understat bloquean requests automatizados.")
        print("  ! Ver instrucciones en el codigo o README sobre como obtener xG gratis.")
        print("  ! Usando goles reales como fallback.\n")

    # Cargar/cachear datos de partidos
    if usar_cache and os.path.exists(cache_file):
        print(">> Usando cache local")
        df = pd.read_parquet(cache_file)

        if usar_cache and os.path.exists(rho_cache):
            with open(rho_cache, 'r') as f:
                rho_por_liga = json.load(f)
        else:
            rho_por_liga = estimar_rho(df)
            with open(rho_cache, 'w') as f:
                json.dump(rho_por_liga, f)

        fuerzas = pd.read_parquet(fuerzas_cache)
        with open(promedios_cache, 'r') as f:
            promedios = json.load(f)

        print(f"  {len(df)} partidos, {len(fuerzas)} equipos\n")
        return LIGAS, fuerzas, promedios, rho_por_liga, xg_data

    # Descargar datos fresh
    print(">> Descargando datos historicos...")
    df = descargar_varias_ligas(LIGAS.keys(), 2425)
    if df.empty:
        df = descargar_varias_ligas(LIGAS.keys(), 2324)
    if df.empty:
        print("X No se pudieron descargar datos")
        return None, None, None, None, None

    print(f"\n  Total: {len(df)} partidos\n")

    # Calcular fuerzas (con xG si esta disponible)
    print(">> Calculando fuerzas...")
    fuerzas, promedios = calcular_fuerzas(df, xg_data=xg_data)
    print(f"  {len(fuerzas)} equipos\n")

    # Estimar rho Dixon-Coles
    print(">> Estimando rho (Dixon-Coles)...")
    rho_por_liga = estimar_rho(df)
    print()

    # Guardar cache
    try:
        df.to_parquet(cache_file)
        fuerzas.to_parquet(fuerzas_cache)
        with open(promedios_cache, 'w') as f:
            json.dump({k: {k2: float(v2) for k2, v2 in v.items()} for k, v in promedios.items()}, f)
        with open(rho_cache, 'w') as f:
            json.dump(rho_por_liga, f)
        print(">> Cache guardado\n")
    except Exception as e:
        print(f"  (cache: {e})\n")

    return LIGAS, fuerzas, promedios, rho_por_liga, xg_data


# ──────────────────────────────────────────────
# 9. OUTPUT
# ──────────────────────────────────────────────

def mostrar_analisis(pred):
    print(f"  Modelo: {pred['modelo']}")
    print(f"  Expected Goals:")
    print(f"    {pred['local']} (local):  {pred['xg_local']:.3f}")
    print(f"    {pred['visitante']} (visit): {pred['xg_visitante']:.3f}")
    print()
    print(f"  Probabilidades:")
    print(f"    {pred['local']}: {pred['prob_local']*100:.1f}%  (cuota justa: {pred['cuota_justa_local']:.2f})")
    print(f"    Empate:    {pred['prob_empate']*100:.1f}%  (cuota justa: {pred['cuota_justa_empate']:.2f})")
    print(f"    {pred['visitante']}: {pred['prob_visitante']*100:.1f}%  (cuota justa: {pred['cuota_justa_visitante']:.2f})")
    print(f"    Over 2.5:  {pred['over_2_5']*100:.1f}%")
    print()
    print(f"  Marcadores mas probables:")
    for sc, p in pred['scorelines'].items():
        print(f"    {sc:5s} -> {p*100:.1f}%")
    print()


def demo(fuerzas, promedios, rho_por_liga):
    print("=" * 65)
    print("  MODELO POISSON + DIXON-COLES")
    print("  Datos: football-data.co.uk + Understat (xG)")
    print("=" * 65)
    print()

    print(">> Top 5 ataque local:")
    top = fuerzas.sort_values('Attack_Strength_Home', ascending=False).head(5)
    for _, e in top.iterrows():
        print(f"  {e['Equipo']:25s}  Ataque local: {e['Attack_Strength_Home']:.3f}")
    print()

    print(">> Rho Dixon-Coles por liga:")
    for liga, rho in rho_por_liga.items():
        print(f"  {LIGAS.get(liga, liga):20s}  rho = {float(rho):.4f}")
    print()

    print(">> Simulacion: Man City vs Arsenal")
    rho_raw = rho_por_liga.get('E0') if rho_por_liga else None
    rho = float(rho_raw) if rho_raw is not None else None
    pred = predecir_partido("Man City", "Arsenal", fuerzas, promedios, 'E0', rho=rho)
    if pred:
        mostrar_analisis(pred)
        values = encontrar_value_bets(pred, 2.10, 3.40, 3.60)
        print(">> Value Bets (vs cuotas ejemplo Bet365):")
        for v in values:
            icono = "VALOR+" if v['es_value'] else "     -"
            print(f"  {icono} {v['mercado']:30s}  Casa:{v['cuota_casa']:.2f}  Justa:{v['cuota_justa']:.2f}  EV:{v['ev']*100:+.1f}%")
        print()
        print(">> Kelly (25%, bankroll $1000):")
        for v in values:
            if v['es_value']:
                print(f"  -> ${kelly_criterion(v['prob_modelo'], v['cuota_casa'])} a {v['mercado']} @ {v['cuota_casa']:.2f}")

    print()
    print("=" * 65)
    print("  python poisson_model.py \"Barcelona vs Real Madrid\"")
    print("  python poisson_model.py --xg \"PSG vs Marseille\"")
    print("  python poisson_model.py --xg-csv mi_xg.csv \"Liverpool vs Arsenal\"")
    print("  python poisson_model.py --no-dc \"Liverpool vs Arsenal\"")
    print("=" * 65)


def analizar_partido(liga_codigo, local, visitante, fuerzas, promedios, rho_por_liga):
    liga_map = {k: v for k, v in LIGAS.items()}
    print(f"\n{'='*65}")
    print(f"  ANALISIS: {local} vs {visitante} ({liga_map.get(liga_codigo, liga_codigo)})")
    print(f"{'='*65}\n")

    rho_raw = rho_por_liga.get(liga_codigo) if rho_por_liga else None
    rho = float(rho_raw) if rho_raw is not None else None
    pred = predecir_partido(local, visitante, fuerzas, promedios, liga_codigo, rho=rho)
    if not pred:
        print("X Equipos no encontrados en esta liga.")
        return
    mostrar_analisis(pred)
    print("  >> Comparamos cuotas justas vs tu casa de apuestas")
    print("     si la casa paga MAS -> hay valor\n")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

if __name__ == '__main__':
    force_refresh = '--refresh' in sys.argv or '-r' in sys.argv
    show_teams = '--teams' in sys.argv or '-t' in sys.argv
    usar_xg = '--xg' in sys.argv
    usar_xg_csv = None
    for i, a in enumerate(sys.argv):
        if a == '--xg-csv' and i + 1 < len(sys.argv):
            usar_xg_csv = sys.argv[i + 1]
            usar_xg = True
    usar_dc = '--no-dc' not in sys.argv  # DC activado por defecto

    args = [a for a in sys.argv[1:] if not a.startswith('-')]

    # Cargar datos
    ligas, fuerzas, promedios, rho_por_liga, xg_data = cargar_datos(
        force_download=force_refresh, usar_xg=usar_xg, xg_csv=usar_xg_csv
    )
    if fuerzas is None:
        sys.exit(1)

    if not usar_dc:
        rho_por_liga = {k: None for k in rho_por_liga}

    if show_teams:
        print("Equipos disponibles por liga:")
        for codigo in ligas:
            print(f"\n  {LIGAS[codigo]}:")
            for eq in sorted(fuerzas[fuerzas['Liga'] == codigo]['Equipo'].unique()):
                print(f"    - {eq}")
        sys.exit(0)

    if args:
        entrada = ' '.join(args)
        parts = re.split(r'\s+vs\s+|,', entrada, maxsplit=1)
        if len(parts) == 2 and parts[1].strip():
            local = resolver_equipo(parts[0].strip())
            visitante = resolver_equipo(parts[1].strip())

            for codigo in ligas:
                eq1 = next((e for e in fuerzas[fuerzas['Liga'] == codigo]['Equipo']
                           if local.lower() in e.lower()), None)
                eq2 = next((e for e in fuerzas[fuerzas['Liga'] == codigo]['Equipo']
                           if visitante.lower() in e.lower()), None)
                if eq1 and eq2:
                    analizar_partido(codigo, eq1, eq2, fuerzas, promedios, rho_por_liga)
                    sys.exit(0)

            print("X Equipos no encontrados. Usa --teams para listarlos.")
        else:
            demo(fuerzas, promedios, rho_por_liga)
    else:
        demo(fuerzas, promedios, rho_por_liga)
