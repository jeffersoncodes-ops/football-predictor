"""
Modelo Poisson para análisis de partidos de fútbol
Basado en: attack/defense strength + Poisson distribution + value betting + Kelly Criterion

Fuente de datos: football-data.co.uk (gratis)
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from math import exp, factorial, log
from datetime import datetime, timedelta
import re
import os
import json

# Atajos comunes de nombres de equipos
TEAM_ALIASES = {
    'manchester city': 'Man City',
    'manchester united': 'Man United',
    'man utd': 'Man United',
    'newcastle united': 'Newcastle',
    'tottenham hotspur': 'Tottenham',
    'spurs': 'Tottenham',
    'nottingham forest': "Nott'm Forest",
    'notts forest': "Nott'm Forest",
    'bayern': 'Bayern Munich',
    'bayern munchen': 'Bayern Munich',
    'eintracht frankfurt': 'Ein Frankfurt',
    'borussia dortmund': 'Dortmund',
    'borussia mgladbach': "M'gladbach",
    'gladbach': "M'gladbach",
    'rb leipzig': 'RB Leipzig',
    'psg': 'Paris SG',
    'paris saint germain': 'Paris SG',
    'inter milan': 'Inter',
    'ac milan': 'Milan',
    'juventus turin': 'Juventus',
    'real betis': 'Betis',
    'athletic bilbao': 'Ath Bilbao',
    'bilbao': 'Ath Bilbao',
    'athletic madrid': 'Ath Madrid',
    'atletico madrid': 'Ath Madrid',
    'atletico': 'Ath Madrid',
    'real sociedad': 'Sociedad',
    'rayo vallecano': 'Vallecano',
    'espanyol': 'Espanol',
    'rcd espanyol': 'Espanol',
    'saint etienne': 'St Etienne',
    'st etienne': 'St Etienne',
    'le havre': 'Le Havre',
    'strasbourg': 'Strasbourg',
}


def resolver_equipo(nombre):
    """Resuelve alias de equipos al nombre exacto usado en football-data.co.uk."""
    n = nombre.lower().strip()
    if n in TEAM_ALIASES:
        return TEAM_ALIASES[n]
    return nombre


# ──────────────────────────────────────────────
# 1. DESCARGA DE DATOS
# ──────────────────────────────────────────────

def descargar_liga(liga, temporada):
    """
    Descarga datos de football-data.co.uk
    liga: 'E0' (Premier), 'SP1' (LaLiga), 'I1' (SerieA), 'D1' (Bundes), 'F1' (Ligue1)
    temporada: 2024 (para 2024-25)
    """
    url = f"https://www.football-data.co.uk/mmz4281/{temporada}/{liga}.csv"
    try:
        df = pd.read_csv(url)
        df = df.copy()
        df.insert(0, 'Liga', liga)
        print(f"  OK {liga} {temporada}/{temporada+1}: {len(df)} partidos")
        return df
    except Exception as e:
        print(f"  XX {liga} {temporada}/{temporada+1}: {e}")
        return None


def descargar_varias_ligas(ligas, temporada):
    """Descarga múltiples ligas y las combina."""
    dfs = []
    for liga in ligas:
        df = descargar_liga(liga, temporada)
        if df is not None:
            dfs.append(df)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


# ──────────────────────────────────────────────
# 2. CÁLCULO DE FUERZAS (ATTACK/DEFENSE STRENGTH)
# ──────────────────────────────────────────────

def calcular_fuerzas(df, min_partidos=5):
    """
    Calcula ataque/defensa local y visitante para cada equipo.
    """
    # Filtrar solo columnas necesarias
    cols_needed = ['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'Liga']
    df = df[[c for c in cols_needed if c in df.columns]].copy()

    df.columns = ['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'Liga']

    # Promedios de la liga
    promedios = df.groupby('Liga').agg(
        home_goles_avg=('FTHG', 'mean'),
        away_goles_avg=('FTAG', 'mean')
    ).to_dict('index')

    # Calcular fuerzas por equipo
    equipos = []

    for liga in df['Liga'].unique():
        df_liga = df[df['Liga'] == liga]
        avg_home = promedios[liga]['home_goles_avg']
        avg_away = promedios[liga]['away_goles_avg']

        # Local
        local = df_liga.groupby('HomeTeam').agg(
            goles_favor=('FTHG', 'sum'),
            goles_contra=('FTAG', 'sum'),
            partidos=('FTHG', 'count')
        ).reset_index()
        local['Liga'] = liga
        local['Tipo'] = 'local'

        # Visitante
        visitante = df_liga.groupby('AwayTeam').agg(
            goles_favor=('FTAG', 'sum'),
            goles_contra=('FTHG', 'sum'),
            partidos=('FTHG', 'count')
        ).reset_index()
        visitante['Liga'] = liga
        visitante['Tipo'] = 'visitante'

        for _, row in local.iterrows():
            if row['partidos'] < min_partidos:
                continue
            equipos.append({
                'Equipo': row['HomeTeam'],
                'Liga': liga,
                'Attack_Strength_Home': row['goles_favor'] / row['partidos'] / avg_home if avg_home > 0 else 1,
                'Defense_Strength_Home': row['goles_contra'] / row['partidos'] / avg_away if avg_away > 0 else 1,
            })

        for _, row in visitante.iterrows():
            if row['partidos'] < min_partidos:
                continue
            # Buscar si ya tenemos el equipo
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
# 3. DISTRIBUCIÓN POISSON
# ──────────────────────────────────────────────

def poisson_prob(lambda_goles, num_goles):
    """Probabilidad de que un equipo meta EXACTAMENTE num_goles goles."""
    if lambda_goles <= 0:
        return 1.0 if num_goles == 0 else 0.0
    return (lambda_goles ** num_goles) * exp(-lambda_goles) / factorial(num_goles)


def calcular_probabilidades_partido(xg_local, xg_visitante, max_goles=10):
    """
    Calcula matriz de probabilidades para todos los marcadores posibles.
    Devuelve: prob_local, prob_empate, prob_visitante, over_2_5, matriz_scorelines
    """
    prob_local = 0
    prob_empate = 0
    prob_visitante = 0
    over_25 = 0
    scorelines = {}

    for g_local in range(max_goles + 1):
        for g_visit in range(max_goles + 1):
            p = poisson_prob(xg_local, g_local) * poisson_prob(xg_visitante, g_visit)
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
# 4. PREDECIR PARTIDO
# ──────────────────────────────────────────────

def predecir_partido(local, visitante, fuerzas, promedios, liga):
    """Predice un partido usando el modelo Poisson."""
    if liga not in promedios:
        return None

    avg_home = promedios[liga]['home_goles_avg']
    avg_away = promedios[liga]['away_goles_avg']

    # Buscar fuerzas
    f_local = fuerzas[(fuerzas['Equipo'] == local) & (fuerzas['Liga'] == liga)]
    f_visit = fuerzas[(fuerzas['Equipo'] == visitante) & (fuerzas['Liga'] == liga)]

    if f_local.empty or f_visit.empty:
        print(f"  ⚠ Datos insuficientes para {local} vs {visitante}")
        return None

    f_local = f_local.iloc[0]
    f_visit = f_visit.iloc[0]

    # Expected Goals
    xg_local = f_local['Attack_Strength_Home'] * f_visit['Defense_Strength_Away'] * avg_home
    xg_visitante = f_visit['Attack_Strength_Away'] * f_local['Defense_Strength_Home'] * avg_away

    prob_local, prob_empate, prob_visit, over_25, scorelines = calcular_probabilidades_partido(
        xg_local, xg_visitante
    )

    return {
        'local': local,
        'visitante': visitante,
        'liga': liga,
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
    }


# ──────────────────────────────────────────────
# 5. VALUE BET DETECTOR
# ──────────────────────────────────────────────

def encontrar_value_bets(prediccion, cuota_local, cuota_empate, cuota_visitante):
    """
    Compara cuotas del modelo vs cuotas de la casa.
    Devuelve las apuestas con valor positivo (value bets).
    """
    results = []

    # Local
    if prediccion['prob_local'] > 0:
        ev_local = (prediccion['prob_local'] * cuota_local) - 1
        results.append({
            'mercado': f"{prediccion['local']} (local)",
            'cuota_casa': cuota_local,
            'cuota_justa': prediccion['cuota_justa_local'],
            'prob_modelo': prediccion['prob_local'],
            'ev': round(ev_local, 4),
            'es_value': ev_local > 0.05  # margen del 5%
        })

    # Empate
    if prediccion['prob_empate'] > 0:
        ev_empate = (prediccion['prob_empate'] * cuota_empate) - 1
        results.append({
            'mercado': 'Empate',
            'cuota_casa': cuota_empate,
            'cuota_justa': prediccion['cuota_justa_empate'],
            'prob_modelo': prediccion['prob_empate'],
            'ev': round(ev_empate, 4),
            'es_value': ev_empate > 0.05
        })

    # Visitante
    if prediccion['prob_visitante'] > 0:
        ev_visit = (prediccion['prob_visitante'] * cuota_visitante) - 1
        results.append({
            'mercado': f"{prediccion['visitante']} (visitante)",
            'cuota_casa': cuota_visitante,
            'cuota_justa': prediccion['cuota_justa_visitante'],
            'prob_modelo': prediccion['prob_visitante'],
            'ev': round(ev_visit, 4),
            'es_value': ev_visit > 0.05
        })

    return results


def kelly_criterion(prob, cuota, bankroll=1000):
    """Calcula el stake óptimo según Kelly Criterion."""
    ev = (prob * cuota) - 1
    if ev <= 0:
        return 0
    q = 1 - prob
    stake_pct = (prob * cuota - 1) / (cuota - 1)
    # Kelly fraccionario (25%) para ser conservadores
    stake_pct = max(0, stake_pct * 0.25)
    return round(bankroll * stake_pct, 2)


# ──────────────────────────────────────────────
# 6. DEMO COMPLETA
# ──────────────────────────────────────────────

def cargar_datos_completos(force_download=False):
    """Carga datos de todas las ligas y calcula fuerzas. Con cache local."""
    ligas = {
        'E0': 'Premier League',
        'SP1': 'LaLiga',
        'I1': 'Serie A',
        'D1': 'Bundesliga',
        'F1': 'Ligue 1',
    }

    cache_file = 'datos_cache.parquet'
    fuerzas_cache = 'fuerzas_cache.parquet'
    promedios_cache = 'promedios_cache.json'

    # Usar cache si existe
    if not force_download and os.path.exists(cache_file):
        print(f">> Usando cache local ({cache_file})")
        df = pd.read_parquet(cache_file)
        fuerzas = pd.read_parquet(fuerzas_cache)
        with open(promedios_cache, 'r') as f:
            promedios = json.load(f)
        print(f"  {len(df)} partidos, {len(fuerzas)} equipos")
        print()
        return ligas, fuerzas, promedios

    print(">> Descargando datos historicos...")
    df = descargar_varias_ligas(ligas.keys(), 2425)
    if df.empty:
        print("  Temporada 2024-25 no disponible, probando 2023-24...")
        df = descargar_varias_ligas(ligas.keys(), 2324)

    if df.empty:
        print("X No se pudieron descargar datos. Verifique conexion.")
        return None, None, None

    print(f"\n  Total: {len(df)} partidos cargados")
    print()
    print(">> Calculando fuerzas de ataque/defensa...")
    fuerzas, promedios = calcular_fuerzas(df)
    print(f"  {len(fuerzas)} equipos analizados")
    print()

    # Guardar cache
    try:
        df.to_parquet(cache_file)
        fuerzas.to_parquet(fuerzas_cache)
        # Convertir promedios a dict serializable
        promedios_serializable = {
            k: {k2: float(v2) for k2, v2 in v.items()}
            for k, v in promedios.items()
        }
        with open(promedios_cache, 'w') as f:
            json.dump(promedios_serializable, f)
        print(f">> Cache guardado. Proximas cargas seran instantaneas.")
        print()
    except Exception as e:
        print(f"  (cache no disponible: {e})")

    return ligas, fuerzas, promedios


def mostrar_analisis(pred, fuerzas):
    """Muestra el analisis completo de una prediccion."""
    print(f"  Expected Goals:")
    print(f"    {pred['local']} (local):  {pred['xg_local']:.3f}")
    print(f"    {pred['visitante']} (visit):  {pred['xg_visitante']:.3f}")
    print()
    print(f"  Probabilidades:")
    print(f"    {pred['local']}: {pred['prob_local']*100:.1f}%  (cuota justa: {pred['cuota_justa_local']:.2f})")
    print(f"    Empate:    {pred['prob_empate']*100:.1f}%  (cuota justa: {pred['cuota_justa_empate']:.2f})")
    print(f"    {pred['visitante']}: {pred['prob_visitante']*100:.1f}%  (cuota justa: {pred['cuota_justa_visitante']:.2f})")
    print(f"    Over 2.5:  {pred['over_2_5']*100:.1f}%")
    print()
    print(f"  Marcadores mas probables:")
    for score, prob in pred['scorelines'].items():
        print(f"    {score:5s} -> {prob*100:.1f}%")
    print()


def demo(fuerzas, promedios):
    """Corre la demo con datos ya cargados."""
    print("=" * 65)
    print("  MODELO POISSON PARA ANALISIS DE PARTIDOS")
    print("  Datos: football-data.co.uk")
    print("=" * 65)
    print()

    # Top ataques locales
    print(">> Top 5 equipos con mejor ataque local:")
    top_ataque = fuerzas.sort_values('Attack_Strength_Home', ascending=False).head(5)
    for _, eq in top_ataque.iterrows():
        print(f"  {eq['Equipo']:25s}  Ataque local: {eq['Attack_Strength_Home']:.3f}")
    print()

    # Simulacion ejemplo
    print(">> Simulacion: Manchester City vs Arsenal")
    pred = predecir_partido("Man City", "Arsenal", fuerzas, promedios, 'E0')
    if pred:
        mostrar_analisis(pred, fuerzas)

        # Value bets
        print(">> Deteccion de Value Bets (vs cuotas ejemplo Bet365):")
        values = encontrar_value_bets(pred, 2.10, 3.40, 3.60)
        for v in values:
            icono = "VALOR+" if v['es_value'] else "     -"
            print(f"  {icono} {v['mercado']:30s}  Casa:{v['cuota_casa']:.2f}  Justa:{v['cuota_justa']:.2f}  EV:{v['ev']*100:+.1f}%")

        print()
        print(">> Stake recomendado (Kelly 25%, bankroll $1000):")
        for v in values:
            if v['es_value']:
                stake = kelly_criterion(v['prob_modelo'], v['cuota_casa'])
                print(f"  -> ${stake} a {v['mercado']} @ {v['cuota_casa']:.2f}")

    print()
    print("=" * 65)
    print("  PARA ANALIZAR OTRO PARTIDO:")
    print("  python poisson_model.py \"Barcelona vs Real Madrid\"")
    print("=" * 65)


def analizar_partido(liga_codigo, local, visitante, fuerzas, promedios):
    """Analiza un partido especifico."""
    liga_map = {
        'E0': 'Premier League', 'SP1': 'LaLiga', 'I1': 'Serie A',
        'D1': 'Bundesliga', 'F1': 'Ligue 1',
    }
    print(f"\n{'='*65}")
    print(f"  ANALISIS: {local} vs {visitante} ({liga_map.get(liga_codigo, liga_codigo)})")
    print(f"{'='*65}\n")

    pred = predecir_partido(local, visitante, fuerzas, promedios, liga_codigo)
    if not pred:
        print("X No se pudo generar la prediccion.")
        print("  Verifica que los nombres de los equipos sean correctos.")
        print("  Equipos disponibles en esta liga:")
        for eq in sorted(fuerzas[fuerzas['Liga'] == liga_codigo]['Equipo'].unique()):
            print(f"    - {eq}")
        return

    mostrar_analisis(pred, fuerzas)

    print(">> Para encontrar value bets, compara estas cuotas justas")
    print("   con las que ofrece tu casa de apuestas favorita.")
    print(f"   Si la casa paga MAS que la cuota justa -> hay valor.")
    print()


if __name__ == '__main__':
    # Flags
    force_refresh = '--refresh' in sys.argv or '-r' in sys.argv
    show_teams = '--teams' in sys.argv or '-t' in sys.argv
    # Filtrar flags de args
    args = [a for a in sys.argv[1:] if not a.startswith('-')]

    # Cargar datos
    ligas, fuerzas, promedios = cargar_datos_completos(force_download=force_refresh)
    if fuerzas is None:
        sys.exit(1)

    # Mostrar equipos disponibles
    if show_teams:
        print("Equipos disponibles por liga:")
        for codigo in ligas:
            print(f"\n  {ligas[codigo]}:")
            for eq in sorted(fuerzas[fuerzas['Liga'] == codigo]['Equipo'].unique()):
                print(f"    - {eq}")
        sys.exit(0)

    # Si pasaron equipos como argumento, analizar ese partido
    if args:
        entrada = ' '.join(args)
        if ' vs ' in entrada:
            parts = entrada.split(' vs ')
        elif ',' in entrada:
            parts = entrada.split(',')
        elif '--' not in entrada:
            parts = entrada.split(None, 1) if ' ' in entrada.strip() else [entrada, '']
        else:
            parts = [entrada, '']

        if len(parts) >= 2 and parts[1].strip():
            local = resolver_equipo(parts[0].strip())
            visitante = resolver_equipo(parts[-1].strip())

            # Buscar la liga automaticamente
            for codigo in ligas:
                eq1 = next((e for e in fuerzas[fuerzas['Liga'] == codigo]['Equipo'] if local.lower() in e.lower()), None)
                eq2 = next((e for e in fuerzas[fuerzas['Liga'] == codigo]['Equipo'] if visitante.lower() in e.lower()), None)
                if eq1 and eq2:
                    analizar_partido(codigo, eq1, eq2, fuerzas, promedios)
                    sys.exit(0)

            print(f"No se encontraron los equipos en ninguna liga.")
            print("Usa: python poisson_model.py --teams")
        else:
            demo(fuerzas, promedios)
    else:
        demo(fuerzas, promedios)
