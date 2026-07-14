#!/usr/bin/env python3
"""
train_qsvc_local.py

Entrena un QSVC por zona con FidelityQuantumKernel usando un sampler local.
Genera por zona en ./outputs/zone_{zone_id}/:
 - modelo: modelo_qsvc_zone_{zone_id}.joblib
 - scaler: scaler_qsvc_zone_{zone_id}.joblib
 - estadisticas: estadisticas_entrenamiento.csv
 - imagenes: superposicion_pca.png, clustering_emergente.png, importancia_sensores.png
 - metadata: metadata.json (last_trained_at, trained_on)
 - interpretaciones: interpretaciones.txt

CORREGIDO: Queries adaptadas a la base de datos ART
 - Tablas reales: lecturas_sensor, nodos_sensor, terrenos, indicadores
 - Columnas reales: timestamp_utc, valor, i.codigo
 - zone_id corresponde a terrenos.id
"""

import os
import sys
import json
import logging
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from joblib import dump, load
import mysql.connector
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Qiskit
try:
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit.primitives import Sampler
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    from qiskit_machine_learning.algorithms.classifiers import QSVC
    from qiskit_algorithms.state_fidelities import ComputeUncompute
    QISKIT_AVAILABLE = True
except Exception:
    QISKIT_AVAILABLE = False

# Aer
# Aer
try:
    from qiskit_aer import Aer
    from qiskit import transpile
    from qiskit.visualization import plot_bloch_multivector

    AER_AVAILABLE = True

except Exception as e:
    logging.exception("Error cargando Aer: %s", str(e))
    AER_AVAILABLE = False

# ---------- Config ----------
TRAIN_SIZE = 60
RANDOM_STATE = 42
FEATURE_COLUMNS = [
    "temperatura",
    "humedad",
    "ph",
    "nitrogeno",
    "fosforo",
    "potasio",
    "conductividad",
    "materia_organica",
    "altitud_msnm"
]
OUTPUT_DIR = "outputs"
# ----------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ----------- Helpers -----------

def cargar_env():
    load_dotenv()
    logging.info("Variables de entorno cargadas desde .env")

def conectar_bd():
    conn = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        port=int(os.getenv("DB_PORT", 3306))
    )
    return conn


def obtener_catalogo_cultivos():
    conn = conectar_bd()

    query = """
    SELECT *
    FROM catalogo_cultivos
    """

    df = pd.read_sql(query, conn)

    conn.close()

    return df


def calcular_cultivo_compatible(row, catalogo):

    mejor_score = -1
    mejor_cultivo = None

    for _, cultivo in catalogo.iterrows():

        score = 0
        total = 9

        if cultivo["ph_min"] <= row["ph"] <= cultivo["ph_max"]:
            score += 1

        if cultivo["humedad_min"] <= row["humedad"] <= cultivo["humedad_max"]:
            score += 1

        if cultivo["temperatura_min"] <= row["temperatura"] <= cultivo["temperatura_max"]:
            score += 1

        if cultivo["nitrogeno_min"] <= row["nitrogeno"] <= cultivo["nitrogeno_max"]:
            score += 1

        if cultivo["fosforo_min"] <= row["fosforo"] <= cultivo["fosforo_max"]:
            score += 1

        if cultivo["potasio_min"] <= row["potasio"] <= cultivo["potasio_max"]:
            score += 1

        if cultivo["conductividad_min"] <= row["conductividad"] <= cultivo["conductividad_max"]:
            score += 1

        if cultivo["materia_organica_min"] <= row["materia_organica"] <= cultivo["materia_organica_max"]:
            score += 1

        if cultivo["altitud_min"] <= row["altitud_msnm"] <= cultivo["altitud_max"]:
            score += 1

        compatibilidad = (score / total) * 100

        if compatibilidad > mejor_score:
            mejor_score = compatibilidad
            mejor_cultivo = cultivo["nombre"]

    return mejor_cultivo, mejor_score

def leer_datos(conn, zone_id=None, limit=TRAIN_SIZE * 10):
    """
    Lee lecturas de la BD ART.
    zone_id = terrenos.id
    Retorna DataFrame con columnas: tipo_sensor, valor, fecha_lectura
    """
    if zone_id is None:
        query = """
            SELECT
                i.codigo AS tipo_sensor,
                l.valor,
                l.timestamp_utc AS fecha_lectura,
                t.altitud_msnm
            FROM lecturas_sensor l
            JOIN nodos_sensor ns ON l.nodo_id = ns.id
            JOIN terrenos t ON ns.terreno_id = t.id
            JOIN indicadores i ON l.indicador_id = i.id
            ORDER BY l.timestamp_utc ASC
            LIMIT %s
        """
        df = pd.read_sql(query, conn, params=(limit,))
    else:
        query = """
            SELECT
                i.codigo AS tipo_sensor,
                l.valor,
                l.timestamp_utc AS fecha_lectura,
                t.altitud_msnm
            FROM lecturas_sensor l
            JOIN nodos_sensor ns ON l.nodo_id = ns.id
            JOIN terrenos t ON ns.terreno_id = t.id
            JOIN indicadores i ON l.indicador_id = i.id
            WHERE t.id = %s
            ORDER BY l.timestamp_utc ASC
            LIMIT %s
        """
        df = pd.read_sql(query, conn, params=(zone_id, limit))

    logging.info("Registros leídos (zone=%s): %d", str(zone_id), len(df))
    return df

def preparar_dataset(df, zone_id):
    if df is None or df.empty:
        return None, None, None

    df["fecha_lectura"] = pd.to_datetime(df["fecha_lectura"])
    df["fecha_minuto"] = df["fecha_lectura"].dt.floor("1min")

    # Normalizar a minúsculas para unificar
    df["tipo_sensor"] = df["tipo_sensor"].str.upper()

    # Mapeo de códigos BD -> nombres FEATURE_COLUMNS
    rename_map = {
        "PH": "ph",
        "NITROGENO": "nitrogeno",
        "FOSFORO": "fosforo",
        "POTASIO": "potasio",
        "HUMEDAD": "humedad",
        "TEMPERATURA": "temperatura",
        "CONDUCTIVIDAD": "conductividad",
        "MATERIA_ORGANICA": "materia_organica"
    }
    df["tipo_sensor"] = df["tipo_sensor"].map(rename_map).fillna(df["tipo_sensor"].str.lower())

    df_pivot = df.pivot_table(
        index="fecha_minuto",
        columns="tipo_sensor",
        values="valor",
        aggfunc="mean"
    ).reset_index()



    print("\n===== PRIMERAS 30 LECTURAS =====")
    print(df[[
        "fecha_lectura",
        "tipo_sensor",
        "valor"
    ]].head(30))

    print("\n===== FECHA MINIMA =====")
    print(df["fecha_lectura"].min())

    print("\n===== FECHA MAXIMA =====")
    print(df["fecha_lectura"].max())

    print("\n===== FECHAS DISTINTAS =====")
    print(df["fecha_lectura"].nunique())


    df_pivot["altitud_msnm"] = df["altitud_msnm"].iloc[0]

    for col in FEATURE_COLUMNS:
        if col not in df_pivot.columns:
            df_pivot[col] = np.nan

    df_clean = df_pivot.dropna(how="all", subset=FEATURE_COLUMNS).copy()

    for col in FEATURE_COLUMNS:

        if df_clean[col].isna().all():

            raise ValueError(
                f"El sensor '{col}' no tiene datos."
            )


    df_clean[FEATURE_COLUMNS] = (
        df_clean[FEATURE_COLUMNS]
        .fillna(df_clean[FEATURE_COLUMNS].mean())
        .fillna(0)
    )

    df_clean = df_clean.head(TRAIN_SIZE)
    if df_clean.shape[0] == 0:
        raise ValueError("❌ No hay datos suficientes después del pivot y limpieza.")

    catalogo = obtener_catalogo_cultivos()

    resultados = df_clean.apply(
        lambda row: calcular_cultivo_compatible(row, catalogo),
        axis=1
    )

    df_clean["cultivo_objetivo"] = resultados.apply(
        lambda x: x[0]
    )


    # FILTRO DE CALIDAD CIENTÍFICA: eliminar clases con muy pocas muestras.
    # Con menos de 5 muestras es imposible validar el modelo de forma
    # confiable (ni siquiera alcanza para 5-fold CV), y distorsiona tanto
    # las métricas como las gráficas. Se documenta como criterio metodológico
    # en el paper: "se excluyeron cultivos con representación insuficiente
    # en los datos de la zona (<5 muestras)".
    MIN_MUESTRAS_POR_CLASE = 5
    conteo = df_clean["cultivo_objetivo"].value_counts()
    clases_validas = conteo[conteo >= MIN_MUESTRAS_POR_CLASE].index.tolist()
    n_antes = len(df_clean)
    df_clean = df_clean[df_clean["cultivo_objetivo"].isin(clases_validas)].copy()
    print(f"\nFiltro de clase minima ({MIN_MUESTRAS_POR_CLASE} muestras): "
          f"{n_antes} -> {len(df_clean)} filas. Clases finales: {clases_validas}")

    if df_clean["cultivo_objetivo"].nunique() < 2:
        raise ValueError(
            "Despues del filtro de clase minima queda menos de 2 cultivos "
            "con datos suficientes. Genera mas lecturas con "
            "06_poblar_lecturas_sensor.py antes de reentrenar."
        )

    print("\n===== CULTIVOS =====")
    print(
        df_clean["cultivo_objetivo"]
        .value_counts()
    )

    df_clean["score_compatibilidad"] = resultados.apply(
        lambda x: x[1]
    )


    X = df_clean[FEATURE_COLUMNS].astype(float).values


    from sklearn.preprocessing import LabelEncoder


    print("\n====== CULTIVOS ======")

    print(
        df_clean["cultivo_objetivo"]
        .value_counts()
    )

    encoder = LabelEncoder()

    y = encoder.fit_transform(
        df_clean["cultivo_objetivo"]
    )


    print("\n=========== CULTIVOS GENERADOS ===========")

    print(
        df_clean[
            [
                "cultivo_objetivo",
                "score_compatibilidad"
            ]
        ].head(100)
    )

    print("\nConteo por cultivo:")

    print(
        df_clean["cultivo_objetivo"].value_counts()
    )

    zone_dir = prepare_zone_dir(zone_id)

    encoder_file = os.path.join(
        zone_dir,
        f"label_encoder_zone_{zone_id}.joblib"
    )

    dump(encoder, encoder_file)




    print("Shape df_pivot:", df_pivot.shape)
    print(df_pivot.head())

    print("Shape X:", X.shape)
    print("Clases únicas:", np.unique(y))
    print("Conteo clases:")
    print(pd.Series(y).value_counts())
    return X, y, df_clean

def escalar_y_guardar(X, zone_dir, zone_id):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    scaler_file = os.path.join(zone_dir, f"scaler_qsvc_zone_{zone_id}.joblib")
    dump(scaler, scaler_file)
    logging.info("Scaler guardado en %s", scaler_file)
    return X_scaled, scaler_file, scaler

# ----------- Quantum Helpers -----------

# DESPUÉS — con PCA de 4 componentes (9 qubits → 4 qubits = 16x más rápido)
N_QUBITS = 9  # una dimensión = un qubit, sin mezclar sensores vía PCA

def entrenar_qsvc(X_train, y_train):
    if not QISKIT_AVAILABLE:
        raise RuntimeError("Qiskit no está disponible en el entorno.")

    n_features = X_train.shape[1]

    if N_QUBITS < n_features:
        # Solo se reduce si decides usar MENOS qubits que sensores
        pca = PCA(n_components=N_QUBITS, random_state=RANDOM_STATE)
        X_reduced = pca.fit_transform(X_train)
    else:
        # N_QUBITS == 9: cada qubit corresponde 1 a 1 a un sensor físico:
        # q0=temperatura, q1=humedad, q2=ph, q3=nitrogeno, q4=fosforo,
        # q5=potasio, q6=conductividad, q7=materia_organica, q8=altitud_msnm
        pca = None
        X_reduced = X_train

    angle_scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_angle = angle_scaler.fit_transform(X_reduced)

    feature_map = ZZFeatureMap(feature_dimension=N_QUBITS, reps=2)
    sampler = Sampler()
    fidelity = ComputeUncompute(sampler)
    qkernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)
    model = QSVC(quantum_kernel=qkernel, probability=True)
    model.fit(X_angle, y_train)
    return model, qkernel, feature_map, pca, angle_scaler


def aplicar_pca_si_existe(pca, X):
    """pca puede ser None cuando N_QUBITS == 9 (sin reducción dimensional)."""
    return pca.transform(X) if pca is not None else X

# ----------- Output Helpers -----------

def asegurar_dir(path):
    os.makedirs(path, exist_ok=True)

def generar_estadisticas(df, y, filename):
    df_stats = df[FEATURE_COLUMNS].describe().T
    df_stats["clase_media"] = pd.Series(y).mean()
    df_stats.to_csv(filename, index=True)
    logging.info("📊 Estadísticas guardadas en %s", filename)

def graficar_superposicion(X_scaled, y, filename, encoder=None):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(7, 5.5))

    if encoder is not None:
        # Un color + etiqueta legible por cada cultivo real, no numeros
        clases_unicas = np.unique(y)
        cmap = matplotlib.colormaps.get_cmap("tab10")
        for i, clase in enumerate(clases_unicas):
            mask = y == clase
            nombre_cultivo = encoder.inverse_transform([clase])[0]
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                        color=cmap(i), alpha=0.75, s=60,
                        label=f"{nombre_cultivo} (n={mask.sum()})")
        plt.legend(title="Cultivo recomendado", loc="best", fontsize=9)
    else:
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="tab10", alpha=0.7)
        plt.colorbar(scatter, label="Clase")

    plt.title("Superposicion cuantica del cultivo (proyeccion PCA 2D)")
    plt.xlabel("Componente Principal 1 (combinacion de sensores)")
    plt.ylabel("Componente Principal 2 (combinacion de sensores)")
    plt.figtext(0.5, -0.02,
                "Cada punto es una lectura de sensor. Puntos del mismo color "
                "= mismo cultivo recomendado por el modelo. Puntos cercanos "
                "entre si = condiciones de suelo similares.",
                ha="center", fontsize=8, wrap=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info("Grafico PCA guardado en %s", filename)

def graficar_clustering(X_scaled, filename, y=None, encoder=None, n_clusters_forzado=None):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    n_clusters = n_clusters_forzado if n_clusters_forzado else min(3, len(X_pca))

    if n_clusters < 2:
        logging.warning("Muy pocas muestras para clustering.")
        return

    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    clusters = kmeans.fit_predict(X_pca)

    plt.figure(figsize=(7, 5.5))
    cmap = matplotlib.colormaps.get_cmap("Set2")
    for c in range(n_clusters):
        mask = clusters == c
        # Si tenemos las etiquetas reales, mostramos que cultivo predomina en cada grupo
        etiqueta = f"Grupo {c+1} (n={mask.sum()})"
        if y is not None and encoder is not None:
            cultivos_del_grupo = y[mask]
            if len(cultivos_del_grupo) > 0:
                dominante = np.bincount(cultivos_del_grupo).argmax()
                nombre = encoder.inverse_transform([dominante])[0]
                etiqueta = f"Grupo {c+1}: predomina '{nombre}' (n={mask.sum()})"
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], color=cmap(c),
                    alpha=0.75, s=60, label=etiqueta)

    plt.legend(loc="best", fontsize=8)
    plt.title("Agrupamiento natural de las lecturas de suelo (K-Means)")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.figtext(0.5, -0.02,
                "Agrupa lecturas con condiciones de suelo parecidas, sin usar "
                "la etiqueta de cultivo. Si los grupos coinciden con los "
                "cultivos reales, confirma que los sensores separan bien "
                "las condiciones de cada cultivo.",
                ha="center", fontsize=8, wrap=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info("Clustering guardado en %s", filename)

def graficar_importancia_sensores(X_scaled, y, filename):
    df_x = pd.DataFrame(X_scaled, columns=FEATURE_COLUMNS)
    y_series = pd.Series(y)
    corr = {}
    for col in FEATURE_COLUMNS:
        if df_x[col].std() == 0:
            corr[col] = 0.0  # sin variacion en esta zona (ej. altitud fija)
        else:
            corr[col] = df_x[col].corr(y_series)
    corr_series = pd.Series(corr).sort_values()
    colores = ["#d62728" if v < 0 else "#2ca02c" for v in corr_series.values]

    plt.figure(figsize=(9, 5.5))
    plt.barh(corr_series.index, corr_series.values, color=colores)
    plt.xlabel("Correlacion con el cultivo recomendado (-1 a +1)")
    plt.title("Que tan influyente es cada sensor en la recomendacion del modelo")
    plt.figtext(0.5, -0.02,
                "Verde = a mayor valor del sensor, tiende a favorecer el cultivo "
                "codificado con numero mas alto. Rojo = relacion inversa. "
                "Barra en cero = el sensor no tuvo variacion en esta zona.",
                ha="center", fontsize=8, wrap=True)
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info("Importancia sensores guardada en %s", filename)
def generar_bloch_image(feature_map, sample_vector, filename):

    if not (QISKIT_AVAILABLE and AER_AVAILABLE):
        logging.warning("Aer o Qiskit no disponibles")
        return None

    try:

        circuit = feature_map.assign_parameters(sample_vector)

        backend = Aer.get_backend("statevector_simulator")

        qc_transpiled = transpile(circuit, backend)

        result = backend.run(qc_transpiled).result()

        state = result.get_statevector()

        fig = plot_bloch_multivector(state)

        fig.savefig(filename, dpi=300, bbox_inches="tight")

        plt.close(fig)

        logging.info("Bloch guardado en %s", filename)

        return filename

    except Exception as e:
        logging.exception("Bloch falló: %s", e)
        return None
# ----------- Interpretaciones -----------

def interpretar_superposicion(pca_file):
    return f"""
🌌 Interpretación de {os.path.basename(pca_file)}:
- Colores separados → modelo distingue condiciones.
- Colores mezclados → datos se solapan, recopilar más registros.
"""

def interpretar_clustering(cluster_file):
    return f"""
🌱 Interpretación de {os.path.basename(cluster_file)}:
- Cada color = grupo natural de suelo.
- Si un cluster domina → parcela homogénea.
- Si varios → variabilidad alta.
"""

def interpretar_importancia(importance_file):
    return f"""
📊 Interpretación de {os.path.basename(importance_file)}:
- Barras altas = sensor más influyente.
- Ej: si pH alto → acidez manda.
- Si humedad domina → riego clave.
"""

def interpretar_bloch(bloch_file):
    return f"""
🔵 Interpretación de {os.path.basename(bloch_file)}:
- La esfera Bloch confirma representación cuántica.
- Dispersión amplia → buena separación.
- Concentrado → posible subajuste.
"""

def cultivos_recomendados(values_dict):
    temp = float(values_dict.get("temperatura", np.nan))
    hum = float(values_dict.get("humedad", np.nan))
    ph = float(values_dict.get("ph", np.nan))
    nitr = float(values_dict.get("nitrogeno", np.nan))
    fosf = float(values_dict.get("fosforo", np.nan))
    pot = float(values_dict.get("potasio", np.nan))
    recomendaciones = []

    if not np.isnan(ph) and not np.isnan(hum):
        if ph < 5.5:
            if hum > 50:
                recomendaciones.append("🌱 Papa, camote, café, piña.")
            else:
                recomendaciones.append("🌱 Papa y camote.")
        elif 5.5 <= ph <= 7.5:
            if 45 <= hum <= 65:
                recomendaciones.append("🌱 Maíz, trigo, frijol, hortalizas.")
            elif hum > 65:
                recomendaciones.append("🌱 Arroz, alfalfa, pastos.")
            else:
                recomendaciones.append("🌱 Quinua, papa (con riego).")
        else:
            recomendaciones.append("🌱 Cebada, remolacha, espárrago.")

    nutrient_avg = np.nanmean([nitr, fosf, pot])
    if not np.isnan(nutrient_avg):
        if nutrient_avg < 10:
            recomendaciones.append("⚠️ Nutrientes bajos — leguminosas.")
        elif nutrient_avg < 25:
            recomendaciones.append("💡 Nutrientes moderados — maíz, papa.")
        else:
            recomendaciones.append("✅ Nutrientes altos — tomate, híbridos.")
    return "\n".join(recomendaciones)

def interpretacion_agronomica(values_dict):
    base = []
    temp = float(values_dict.get("temperatura", np.nan))
    hum = float(values_dict.get("humedad", np.nan))
    ph = float(values_dict.get("ph", np.nan))
    nitr = float(values_dict.get("nitrogeno", np.nan))
    fosf = float(values_dict.get("fosforo", np.nan))
    pot = float(values_dict.get("potasio", np.nan))

    if not np.isnan(hum):
        if hum < 40:
            base.append("💧 Riego recomendado: Sí (humedad baja).")
        elif hum < 55:
            base.append("💧 Riego recomendado: Monitorear.")
        else:
            base.append("💧 Riego recomendado: No necesario.")
    if not np.isnan(ph):
        if ph < 5.5:
            base.append("🧪 pH ácido, aplicar cal.")
        elif ph > 7.5:
            base.append("🧪 pH alcalino, usar enmiendas.")
        else:
            base.append("🧪 pH óptimo.")
    nutrient_avg = np.nanmean([nitr, fosf, pot])
    if not np.isnan(nutrient_avg):
        if nutrient_avg < 10:
            base.append("🌾 Nutrientes bajos — fertilizar urgente.")
        elif nutrient_avg < 25:
            base.append("🌾 Nutrientes moderados — fertilización leve.")
        else:
            base.append("🌾 Nutrientes adecuados.")
    base.append("🔍 Cultivos recomendados:")
    base.append(cultivos_recomendados(values_dict))
    return "\n".join(base)

def guardar_interpretaciones(zone_dir, pca_file, cluster_file, importance_file, bloch_file):
    interpretaciones = []
    interpretaciones.append(interpretar_superposicion(pca_file))
    interpretaciones.append(interpretar_clustering(cluster_file))
    interpretaciones.append(interpretar_importancia(importance_file))
    if bloch_file:
        interpretaciones.append(interpretar_bloch(bloch_file))
    with open(os.path.join(zone_dir, "interpretaciones.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(interpretaciones))
    logging.info("📝 Interpretaciones guardadas en interpretaciones.txt")

# -------------- Metadata ----------------

def guardar_metadata(zone_dir, last_data_ts):
    meta = {
        "last_trained_at": last_data_ts.isoformat() if isinstance(last_data_ts, datetime) else str(last_data_ts),
        "trained_on": datetime.utcnow().isoformat()
    }
    with open(os.path.join(zone_dir, "metadata.json"), "w") as f:
        json.dump(meta, f)
    logging.info("Metadata guardada en %s", zone_dir)

# -------------- Entrenamiento por zona --------------

def prepare_zone_dir(zone_id):
    zone_dir = os.path.join(OUTPUT_DIR, f"zone_{zone_id}")
    asegurar_dir(zone_dir)
    return zone_dir

# DESPUÉS — solo cambia None por zone_id
def train_zone(zone_id):
    try:
        cargar_env()
        conn = conectar_bd()
        df = leer_datos(conn, zone_id=zone_id)  # ← CORRECTO
        conn.close()
        if df is None or df.empty:
            raise ValueError("No hay datos para la zona solicitada.")

        X, y, df_clean = preparar_dataset(
            df,
            zone_id
        )


        unique_classes = np.unique(y)



        if len(unique_classes) < 2:
            raise ValueError(
                f"No se puede entrenar. Solo existe una clase: {unique_classes}"
            )
        

        if len(y) < 10:
            raise ValueError(
                "Muy pocos registros para entrenar."
            )
        zone_dir = prepare_zone_dir(zone_id)
        X_scaled, scaler_file, scaler = escalar_y_guardar(X, zone_dir, zone_id)

        # DESPUÉS
        import time
        _t0_final = time.time()
        print("Entrenando modelo QSVC FINAL con todos los datos...", flush=True)
        model, qkernel, feature_map, pca_quantum, angle_scaler = entrenar_qsvc(X_scaled, y)
        print(f"Modelo final entrenado en {time.time()-_t0_final:.1f}s", flush=True)

        # Guardar pca y angle_scaler para inferencia futura (AMBOS son necesarios)
        pca_file_q = os.path.join(zone_dir, f"pca_quantum_zone_{zone_id}.joblib")
        dump(pca_quantum, pca_file_q)
        angle_scaler_file = os.path.join(zone_dir, f"angle_scaler_zone_{zone_id}.joblib")
        dump(angle_scaler, angle_scaler_file)

        # Predecir con PCA + escalado angular aplicado
        X_pca_pred = aplicar_pca_si_existe(pca_quantum, X_scaled)
        X_angle_pred = angle_scaler.transform(X_pca_pred)
        y_pred = model.predict(X_angle_pred)


        model_file = os.path.join(zone_dir, f"modelo_qsvc_zone_{zone_id}.joblib")
        dump(model, model_file)
        logging.info("✅ Modelo guardado en %s", model_file)

        # DESPUÉS — validación cruzada + baselines comparativos
        from sklearn.svm import SVC
        from sklearn.model_selection import StratifiedKFold, cross_val_score

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        # ── Baseline 1: SVM clásico RBF ──
        svm_rbf = SVC(kernel='rbf', C=1.0, random_state=RANDOM_STATE)
        scores_rbf = cross_val_score(svm_rbf, X_scaled, y, cv=skf, scoring='accuracy')

        # ── Baseline 2: SVM polinomial ──
        svm_poly = SVC(kernel='poly', degree=3, random_state=RANDOM_STATE)
        scores_poly = cross_val_score(svm_poly, X_scaled, y, cv=skf, scoring='accuracy')

        # ── Entrenar y GUARDAR el SVM-RBF final con probabilidades habilitadas ──
        # (esto es lo que permite comparar clásico vs cuántico EN VIVO desde /predict,
        # no solo en el CSV del paper)
        svm_rbf_final = SVC(kernel='rbf', C=1.0, random_state=RANDOM_STATE, probability=True)
        svm_rbf_final.fit(X_scaled, y)
        svm_rbf_file = os.path.join(zone_dir, f"svm_rbf_zone_{zone_id}.joblib")
        dump(svm_rbf_final, svm_rbf_file)
        logging.info("SVM-RBF final guardado en %s", svm_rbf_file)

        # ── Modelo propuesto: QSVC con UN SOLO hold-out (no 5-fold completo) ──
        # NOTA METODOLÓGICA para el paper: por el costo computacional O(n^2)
        # de la simulación del kernel cuántico (cada evaluación requiere
        # simular un circuito), se usa un unico hold-out estratificado en
        # lugar de 5-fold completo para el QSVC, manteniendo 5-fold para los
        # baselines clásicos (computacionalmente triviales). Esta es una
        # limitación reportada en la literatura sobre el costo de kernels
        # cuánticos en simuladores actuales (NISQ/simulacion clasica).
        import time
        t0 = time.time()
        print("Entrenando QSVC (hold-out unico, puede tardar varios minutos)...", flush=True)

        from sklearn.model_selection import train_test_split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_scaled, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
        )
        m_holdout, _, _, pca_fold, angle_fold = entrenar_qsvc(X_tr, y_tr)
        X_val_transformed = angle_fold.transform(aplicar_pca_si_existe(pca_fold, X_val))
        acc_qsvc_mean = accuracy_score(y_val, m_holdout.predict(X_val_transformed))
        acc_qsvc_std = 0.0  # no aplica con un solo split

        print(f"QSVC hold-out listo en {time.time()-t0:.1f}s -> acc={acc_qsvc_mean:.3f}", flush=True)

        logging.info("SVM-RBF  : %.3f ± %.3f", scores_rbf.mean(),  scores_rbf.std())
        logging.info("SVM-Poly : %.3f ± %.3f", scores_poly.mean(), scores_poly.std())
        logging.info("QSVC     : %.3f ± %.3f", acc_qsvc_mean, acc_qsvc_std)



        # Tabla comparativa → directo al CSV del paper
        tabla_paper = pd.DataFrame({
            "Modelo":         ["SVM-RBF",          "SVM-Poly",          "QSVC-ZZFeatureMap"],
            "Accuracy_mean":  [scores_rbf.mean(),  scores_poly.mean(),  acc_qsvc_mean],
            "Accuracy_std":   [scores_rbf.std(),   scores_poly.std(),   acc_qsvc_std],
            "N_muestras":     [len(y), len(y), len(y)],
            "N_features_in":  [X_scaled.shape[1], X_scaled.shape[1], N_QUBITS],
            "Zone_id":        [zone_id, zone_id, zone_id],
        })
        tabla_path = os.path.join(zone_dir, "comparacion_modelos_paper.csv")
        tabla_paper.to_csv(tabla_path, index=False, float_format="%.4f")
        logging.info("Tabla paper guardada en %s", tabla_path)

        stats_file = os.path.join(zone_dir, "estadisticas_entrenamiento.csv")
        pca_file = os.path.join(zone_dir, "superposicion_pca.png")
        cluster_file = os.path.join(zone_dir, "clustering_emergente.png")
        importancia_file = os.path.join(zone_dir, "importancia_sensores.png")
        bloch_file = os.path.join(zone_dir, "bloch_superposicion.png")

        generar_estadisticas(df_clean, y, stats_file)
        encoder_path_actual = os.path.join(zone_dir, f"label_encoder_zone_{zone_id}.joblib")
        encoder_actual = load(encoder_path_actual)
        graficar_superposicion(X_scaled, y, pca_file, encoder=encoder_actual)
        graficar_clustering(X_scaled, cluster_file, y=y, encoder=encoder_actual,
                             n_clusters_forzado=len(np.unique(y)))
        graficar_importancia_sensores(X_scaled, y, importancia_file)

        try:
            # Reducir sample a N_QUBITS dimensiones igual que en el entrenamiento
            sample_scaled = scaler.transform(np.mean(X, axis=0).reshape(1, -1))
            sample_pca = aplicar_pca_si_existe(pca_quantum, sample_scaled)
            sample_angle = angle_scaler.transform(sample_pca)[0]  # FIX: faltaba este paso
            bloch_path = generar_bloch_image(feature_map, sample_angle, bloch_file)
            if not bloch_path and os.path.exists(bloch_file):
                os.remove(bloch_file)
        except Exception:
            logging.exception("No se pudo generar Bloch image.")
            bloch_path = None

        guardar_interpretaciones(zone_dir, pca_file, cluster_file, importancia_file, bloch_path)

        last_ts = pd.to_datetime(
            df["fecha_lectura"]
        ).max()
        guardar_metadata(zone_dir, last_ts)

        # Calcular recomendación dominante de la zona
        cultivo_dominante = df_clean["cultivo_objetivo"].value_counts().idxmax()
        score_dominante   = df_clean.loc[
            df_clean["cultivo_objetivo"] == cultivo_dominante,
            "score_compatibilidad"
        ].mean()

        valores_medios = df_clean[FEATURE_COLUMNS].mean().to_dict()
        resumen_agronomico = interpretacion_agronomica(valores_medios)

        try:
            guardar_recomendacion(zone_id, cultivo_dominante, score_dominante)
            logging.info("Recomendación guardada en BD: %s (%.1f%%)", cultivo_dominante, score_dominante)
        except Exception as e:
            logging.warning("No se pudo guardar recomendación en BD: %s", e)

        return {
            "model_file": model_file,
            "scaler_file": scaler_file,
            "stats_file": stats_file,
            "pca_file": pca_file,
            "cluster_file": cluster_file,
            "importance_file": importancia_file,
            "bloch_file": bloch_path if bloch_path else None,
            "interpretaciones": os.path.join(zone_dir, "interpretaciones.txt"),
            "last_trained_at": last_ts.isoformat() if last_ts is not None else None,
            # ---- NUEVO: recomendación explícita para el paper ----
            "recomendacion_zona": {
                "cultivo_recomendado": cultivo_dominante,
                "score_compatibilidad_promedio": round(float(score_dominante), 2),
                "resumen_agronomico": resumen_agronomico,
                "distribucion_cultivos": df_clean["cultivo_objetivo"].value_counts().to_dict()
            }
        }
    except Exception as e:
        logging.exception("Error en train_zone: %s", str(e))
        raise

# -------------- Inferencia ----------------

def load_model_for_zone(zone_id):
    zone_dir = prepare_zone_dir(zone_id)
    model_path = os.path.join(zone_dir, f"modelo_qsvc_zone_{zone_id}.joblib")
    scaler_path = os.path.join(zone_dir, f"scaler_qsvc_zone_{zone_id}.joblib")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Modelo o scaler no encontrado.")
    model = load(model_path)
    scaler = load(scaler_path)
    return model, scaler, zone_dir

def predict_from_values(zone_id, values_dict):
    X = np.array([[float(values_dict.get(col, np.nan)) for col in FEATURE_COLUMNS]], dtype=float)
    if np.isnan(X).any():
        raise ValueError("Faltan valores en values_dict.")

    model, scaler, zone_dir = load_model_for_zone(zone_id)
    X_scaled = scaler.transform(X)

    pca_path = os.path.join(zone_dir, f"pca_quantum_zone_{zone_id}.joblib")
    if not os.path.exists(pca_path):
        raise FileNotFoundError(f"PCA cuántico no encontrado: {pca_path}")
    pca_quantum = load(pca_path)
    X_pca = aplicar_pca_si_existe(pca_quantum, X_scaled)

    # FIX: aplicar el MISMO escalado angular que se usó al entrenar
    angle_scaler_path = os.path.join(zone_dir, f"angle_scaler_zone_{zone_id}.joblib")
    if not os.path.exists(angle_scaler_path):
        raise FileNotFoundError(
            f"angle_scaler no encontrado: {angle_scaler_path}. "
            "Reentrena la zona (botón 'Entrenar modelo') para regenerarlo."
        )
    angle_scaler = load(angle_scaler_path)
    X_angle = angle_scaler.transform(X_pca)

    encoder_file = os.path.join(zone_dir, f"label_encoder_zone_{zone_id}.joblib")
    encoder = load(encoder_file)

    pred_num = model.predict(X_angle)[0]
    cultivo = encoder.inverse_transform([pred_num])[0]

    # Probabilidades del modelo CUÁNTICO por cultivo (%)
    proba_q = model.predict_proba(X_angle)[0]
    probabilidades_qsvc = {
        str(encoder.inverse_transform([i])[0]): round(float(p) * 100, 1)
        for i, p in enumerate(proba_q)
    }

    # Comparación con el modelo CLÁSICO (SVM-RBF) -> esto es tu demostración en vivo
    comparacion_clasica = None
    svm_rbf_path = os.path.join(zone_dir, f"svm_rbf_zone_{zone_id}.joblib")
    if os.path.exists(svm_rbf_path):
        svm_rbf = load(svm_rbf_path)
        pred_svm_num = svm_rbf.predict(X_scaled)[0]
        proba_svm = svm_rbf.predict_proba(X_scaled)[0]
        cultivo_svm = encoder.inverse_transform([pred_svm_num])[0]
        probabilidades_svm = {
            str(encoder.inverse_transform([i])[0]): round(float(p) * 100, 1)
            for i, p in enumerate(proba_svm)
        }
        comparacion_clasica = {
            "cultivo_svm_rbf": str(cultivo_svm),
            "probabilidades_svm_rbf": probabilidades_svm,
            "coinciden_clasico_y_cuantico": bool(cultivo_svm == cultivo),
        }

    return cultivo, X_scaled, zone_dir, probabilidades_qsvc, comparacion_clasica




def analizar_sensibilidad(zone_id, values_dict, paso_pct=0.15):
    """
    Para cada sensor, calcula cuánto cambia la probabilidad del cultivo
    dominante si ese sensor sube o baja un 15%. Sirve para responder:
    '¿qué sensor debo mover, y cuánto, para acercarme al otro cultivo?'
    """
    model, scaler, zone_dir = load_model_for_zone(zone_id)
    pca_quantum = load(os.path.join(zone_dir, f"pca_quantum_zone_{zone_id}.joblib"))
    angle_scaler = load(os.path.join(zone_dir, f"angle_scaler_zone_{zone_id}.joblib"))
    encoder = load(os.path.join(zone_dir, f"label_encoder_zone_{zone_id}.joblib"))

    base_vals = {c: float(values_dict.get(c, np.nan)) for c in FEATURE_COLUMNS}
    if any(np.isnan(v) for v in base_vals.values()):
        raise ValueError("Faltan valores en values_dict.")
    



    

    def proba_para(vals):
        X = np.array([[vals[c] for c in FEATURE_COLUMNS]], dtype=float)
        X_scaled = scaler.transform(X)
        X_pca = aplicar_pca_si_existe(pca_quantum, X_scaled)
        X_angle = angle_scaler.transform(X_pca)
        proba = model.predict_proba(X_angle)[0]
        return {str(encoder.inverse_transform([i])[0]): float(p) for i, p in enumerate(proba)}

    proba_base = proba_para(base_vals)
    cultivo_base = max(proba_base, key=proba_base.get)
    # Ahora el objetivo es reforzar el cultivo YA recomendado, no el alternativo
    cultivo_objetivo = cultivo_base

    resultados = []
    for sensor in FEATURE_COLUMNS:
        val_actual = base_vals[sensor]

        vals_sube = dict(base_vals)
        vals_sube[sensor] = val_actual * (1 + paso_pct)
        proba_sube = proba_para(vals_sube)

        vals_baja = dict(base_vals)
        vals_baja[sensor] = val_actual * (1 - paso_pct)
        proba_baja = proba_para(vals_baja)

        delta_sube = proba_sube[cultivo_objetivo] - proba_base[cultivo_objetivo]
        delta_baja = proba_baja[cultivo_objetivo] - proba_base[cultivo_objetivo]

        if delta_sube >= delta_baja:
            mejor_direccion = "subir"
            mejor_delta = delta_sube
        else:
            mejor_direccion = "bajar"
            mejor_delta = delta_baja
        resultados.append({
            "sensor": sensor,
            "valor_actual": round(val_actual, 2),
            "direccion_recomendada": mejor_direccion,
            "cambio_probabilidad_pct": round(mejor_delta * 100, 2),
        })

    resultados.sort(key=lambda r: r["cambio_probabilidad_pct"], reverse=True)

    return {
        "cultivo_actual": cultivo_base,
        "cultivo_objetivo": cultivo_objetivo,
        "probabilidad_actual_objetivo": round(proba_base[cultivo_objetivo] * 100, 1),
        "sensibilidad_por_sensor": resultados,
    }


def analizar_sensibilidad_todos_cultivos(zone_id, values_dict, paso_pct=0.15):
    """
    Para CADA cultivo del catálogo, calcula qué sensor mover y en qué
    dirección para acercarse a ese cultivo. Reemplaza el bloque
    hardcodeado de interpretacion_agronomica/cultivos_recomendados.
    """
    model, scaler, zone_dir = load_model_for_zone(zone_id)
    pca_quantum = load(os.path.join(zone_dir, f"pca_quantum_zone_{zone_id}.joblib"))
    angle_scaler = load(os.path.join(zone_dir, f"angle_scaler_zone_{zone_id}.joblib"))
    encoder = load(os.path.join(zone_dir, f"label_encoder_zone_{zone_id}.joblib"))

    base_vals = {c: float(values_dict.get(c, np.nan)) for c in FEATURE_COLUMNS}
    if any(np.isnan(v) for v in base_vals.values()):
        raise ValueError("Faltan valores en values_dict.")

    def proba_para(vals):
        X = np.array([[vals[c] for c in FEATURE_COLUMNS]], dtype=float)
        X_scaled = scaler.transform(X)
        X_pca = aplicar_pca_si_existe(pca_quantum, X_scaled)
        X_angle = angle_scaler.transform(X_pca)
        proba = model.predict_proba(X_angle)[0]
        return {str(encoder.inverse_transform([i])[0]): float(p) for i, p in enumerate(proba)}

    proba_base = proba_para(base_vals)
    todos_los_cultivos = list(proba_base.keys())
    resultado_por_cultivo = {}

    for cultivo_target in todos_los_cultivos:
        resultados_sensores = []
        for sensor in FEATURE_COLUMNS:
            val_actual = base_vals[sensor]
            vals_sube = dict(base_vals); vals_sube[sensor] = val_actual * (1 + paso_pct)
            proba_sube = proba_para(vals_sube)
            vals_baja = dict(base_vals); vals_baja[sensor] = val_actual * (1 - paso_pct)
            proba_baja = proba_para(vals_baja)

            delta_sube = proba_sube[cultivo_target] - proba_base[cultivo_target]
            delta_baja = proba_baja[cultivo_target] - proba_base[cultivo_target]
            if delta_sube >= delta_baja:
                mejor_direccion, mejor_delta = "subir", delta_sube
            else:
                mejor_direccion, mejor_delta = "bajar", delta_baja

            resultados_sensores.append({
                "sensor": sensor,
                "valor_actual": round(val_actual, 2),
                "direccion_recomendada": mejor_direccion,
                "cambio_probabilidad_pct": round(mejor_delta * 100, 2),
            })

        resultados_sensores.sort(key=lambda r: r["cambio_probabilidad_pct"], reverse=True)
        resultado_por_cultivo[cultivo_target] = {
            "probabilidad_actual_pct": round(proba_base[cultivo_target] * 100, 1),
            "sensor_mas_influyente": resultados_sensores[0],
            "todos_los_sensores": resultados_sensores,
        }

    return resultado_por_cultivo


def generar_interpretacion_simple(cultivo_pred, proba_qsvc, comparacion_clasica, sensibilidad):
    """
    Traduce los resultados numéricos a un texto que cualquier persona
    (sin conocimiento técnico) pueda entender.
    """
    lineas = []

    conf = proba_qsvc.get(cultivo_pred, 0)
    otros = {c: p for c, p in proba_qsvc.items() if c != cultivo_pred}

    # 1. Qué recomienda el modelo y qué tan seguro está
    if conf >= 70:
        seguridad = "con bastante seguridad"
    elif conf >= 55:
        seguridad = "aunque con seguridad moderada"
    else:
        seguridad = "pero la decisión es muy pareja entre cultivos, no es un resultado contundente"

    lineas.append(
        f"El sistema recomienda sembrar {cultivo_pred} {seguridad} "
        f"({conf}% de probabilidad)."
    )

    if otros:
        detalle_otros = ", ".join(f"{c} ({p}%)" for c, p in sorted(otros.items(), key=lambda x: -x[1]))
        lineas.append(f"Los demás cultivos evaluados quedaron en: {detalle_otros}.")

    # 2. Comparación con el modelo clásico
    if comparacion_clasica:
        if comparacion_clasica["coinciden_clasico_y_cuantico"]:
            lineas.append(
                "El modelo clásico (SVM) llegó a la misma recomendación, "
                "lo cual da más confianza al resultado."
            )
        else:
            lineas.append(
                f"Sin embargo, el modelo clásico (SVM) recomienda "
                f"{comparacion_clasica['cultivo_svm_rbf']} en su lugar. "
                "Cuando los dos modelos no coinciden, conviene revisar el terreno "
                "en persona antes de decidir."
            )

    # 3. Qué se podría cambiar para favorecer otro cultivo
    if sensibilidad and sensibilidad.get("sensibilidad_por_sensor"):
        top = sensibilidad["sensibilidad_por_sensor"][0]
        objetivo = sensibilidad["cultivo_objetivo"]
        direccion = "aumentar" if top["direccion_recomendada"] == "subir" else "disminuir"
        lineas.append(
            f"Si se quisiera favorecer al cultivo alternativo ({objetivo}), "
            f"el factor que más influye es {top['sensor']} — habría que "
            f"{direccion}lo respecto a su valor actual ({top['valor_actual']})."
        )

    return " ".join(lineas)

def evaluar_lote_zona(zone_id, n_muestras=None):
    """
    Evalúa TODAS (o las primeras n_muestras) las lecturas de la zona
    con ambos modelos (QSVC y SVM-RBF) y compara contra el cultivo
    real calculado por reglas agronómicas (cultivo_objetivo).
    Esto es la evidencia comparativa real para el paper.
    """
    conn = conectar_bd()
    df = leer_datos(conn, zone_id=zone_id)
    conn.close()

    X, y, df_clean = preparar_dataset(df, zone_id)
    if n_muestras:
        df_clean = df_clean.head(n_muestras)
        X = X[:len(df_clean)]
        y = y[:len(df_clean)]

    model, scaler, zone_dir = load_model_for_zone(zone_id)
    pca_quantum = load(os.path.join(zone_dir, f"pca_quantum_zone_{zone_id}.joblib"))
    angle_scaler = load(os.path.join(zone_dir, f"angle_scaler_zone_{zone_id}.joblib"))
    encoder = load(os.path.join(zone_dir, f"label_encoder_zone_{zone_id}.joblib"))
    svm_rbf = load(os.path.join(zone_dir, f"svm_rbf_zone_{zone_id}.joblib"))

    X_scaled = scaler.transform(X)
    X_pca = aplicar_pca_si_existe(pca_quantum, X_scaled)
    X_angle = angle_scaler.transform(X_pca)

    pred_qsvc = model.predict(X_angle)
    proba_qsvc = model.predict_proba(X_angle)
    pred_svm = svm_rbf.predict(X_scaled)
    proba_svm = svm_rbf.predict_proba(X_scaled)

    filas = []
    for i in range(len(y)):
        cultivo_real = encoder.inverse_transform([y[i]])[0]
        cultivo_qsvc = encoder.inverse_transform([pred_qsvc[i]])[0]
        cultivo_svm = encoder.inverse_transform([pred_svm[i]])[0]
        filas.append({
            "muestra": i + 1,
            "cultivo_real": str(cultivo_real),
            "qsvc_pred": str(cultivo_qsvc),
            "qsvc_confianza": round(float(max(proba_qsvc[i])) * 100, 1),
            "svm_pred": str(cultivo_svm),
            "svm_confianza": round(float(max(proba_svm[i])) * 100, 1),
            "qsvc_acierto": bool(cultivo_qsvc == cultivo_real),
            "svm_acierto": bool(cultivo_svm == cultivo_real),
            "coinciden_modelos": bool(cultivo_qsvc == cultivo_svm),
        })

    resumen = {
        "n_muestras": len(filas),
        "accuracy_qsvc": round(sum(f["qsvc_acierto"] for f in filas) / len(filas) * 100, 1),
        "accuracy_svm": round(sum(f["svm_acierto"] for f in filas) / len(filas) * 100, 1),
        "pct_coincidencia_modelos": round(sum(f["coinciden_modelos"] for f in filas) / len(filas) * 100, 1),
    }

    return {"resumen": resumen, "detalle": filas}


def guardar_recomendacion(
        terreno_id,
        cultivo_nombre,
        score):

    conn = conectar_bd()

    cur = conn.cursor()

    cur.execute("""
        SELECT id
        FROM catalogo_cultivos
        WHERE nombre=%s
    """, (cultivo_nombre,))

    resultado = cur.fetchone()

    if resultado is None:
        raise ValueError(
            f"Cultivo no encontrado: {cultivo_nombre}"
        )

    cultivo_id = resultado[0]


    if score >= 80:
        calidad = 5
    elif score >= 60:
        calidad = 4
    elif score >= 40:
        calidad = 3
    elif score >= 20:
        calidad = 2
    else:
        calidad = 1

    cur.execute("""
    INSERT INTO recomendaciones_cultivo(
        terreno_id,
        cultivo_id,
        calidad_suelo_id,
        score_compatibilidad
    )
    VALUES (%s,%s,%s,%s)
    """, (
        int(terreno_id),
        int(cultivo_id),
        int(calidad),
        float(score)   # float nativo de Python, no numpy float64
    ))

    conn.commit()

    conn.close()

# -------------- MAIN ----------------

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        zid = sys.argv[1]
        out = train_zone(zid)
        print("Entrenamiento completado:", out)
    else:
        print("Uso: python train_qsvc_local.py <zone_id>")