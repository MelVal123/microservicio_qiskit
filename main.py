#!/usr/bin/env python3
"""
main.py - API con FastAPI

Endpoints:
- POST /train        -> Entrena (o reentrena) una zona: {"zone_id": <int>}
- POST /predict      -> Predice para una zona.
- Static files: /outputs/... (imágenes/CSV generados)
"""

import os
import logging
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from datetime import datetime
import json
import mysql.connector
from dotenv import load_dotenv

import train_qsvc_local as trainer

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

app = FastAPI(title="🌱 Quantum Agriculture API", version="1.0")

# ----------------- CORS -----------------
# 🚨 IMPORTANTE: incluimos el mismo dominio del backend y permitimos Swagger UI
origins = [
    "*",  # Para pruebas (puedes luego restringirlo)
    "http://localhost:5173",
    "http://localhost:3000",
    "https://zonas.grupo-digital-nextri.com",
    "https://qiskit-production.up.railway.app",
    "https://microservicioqiskit-production.up.railway.app",  # <--- Tu backend actual
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Static files -----------------
if not os.path.exists("outputs"):
    os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# ----------------- Modelos -----------------
class TrainRequest(BaseModel):
    zone_id: int

# ----------------- DB -----------------
def conectar_bd():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        port=int(os.getenv("DB_PORT", 3306))
    )
def obtener_ultimas_lecturas_de_zona(zone_id):
    conn = conectar_bd()
    cursor = conn.cursor(dictionary=True)
    
    # ✅ Traer lecturas de sensores
    query = """
        SELECT
            i.codigo AS sensor,
            l.valor,
            l.timestamp_utc AS fecha_lectura
        FROM lecturas_sensor l
        JOIN nodos_sensor ns ON l.nodo_id = ns.id
        JOIN terrenos t ON ns.terreno_id = t.id
        JOIN indicadores i ON l.indicador_id = i.id
        WHERE t.id = %s
        ORDER BY l.timestamp_utc DESC
    """
    cursor.execute(query, (zone_id,))
    rows = cursor.fetchall()

    # ✅ Traer altitud_msnm del terreno
    cursor.execute("""
        SELECT altitud_msnm FROM terrenos WHERE id = %s
    """, (zone_id,))
    terreno = cursor.fetchone()
    conn.close()

    if not rows:
        return {}

    result = {}
    for row in rows:
        key = row["sensor"]
        if key not in result:
            result[key] = {"valor": float(row["valor"]), "fecha_lectura": row["fecha_lectura"]}

    ren = {
        "PH": "ph",
        "NITROGENO": "nitrogeno",
        "FOSFORO": "fosforo",
        "POTASIO": "potasio",
        "HUMEDAD": "humedad",
        "TEMPERATURA": "temperatura",
        "CONDUCTIVIDAD": "conductividad",
        "MATERIA_ORGANICA": "materia_organica"
    }
    result = {ren.get(k, k): v for k, v in result.items()}

    # ✅ Agregar altitud_msnm al resultado
    if terreno and terreno["altitud_msnm"] is not None:
        result["altitud_msnm"] = {"valor": float(terreno["altitud_msnm"]), "fecha_lectura": None}

    return result
def obtener_max_fecha_lectura(zone_id):
    conn = conectar_bd()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT MAX(l.timestamp_utc)
        FROM lecturas_sensor l
        JOIN nodos_sensor ns ON l.nodo_id = ns.id
        JOIN terrenos t ON ns.terreno_id = t.id
        WHERE t.id = %s
    """, (zone_id,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None


# ----------------- Endpoints -----------------
@app.post("/train")
def train(req: TrainRequest):
    try:
        out = trainer.train_zone(req.zone_id)
        base = f"/outputs/zone_{req.zone_id}/"
        files = {
            "model": base + os.path.basename(out["model_file"]),
            "scaler": base + os.path.basename(out["scaler_file"]),
            "stats_csv": base + os.path.basename(out["stats_file"]),
            "pca_png": base + os.path.basename(out["pca_file"]),
            "cluster_png": base + os.path.basename(out["cluster_file"]),
            "importance_png": base + os.path.basename(out["importance_file"]),
            "bloch_png": (
                base + os.path.basename(out["bloch_file"])
                if out.get("bloch_file")
                else None
            ),
            "last_trained_at": out["last_trained_at"]
        }
        return {"status": "ok", "zone_id": req.zone_id, "files": files}
    except Exception as e:
        logging.exception("Error en /train")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict(raw_body: Dict[str, Any] = Body(...), zone_id: Optional[int] = Query(None)):
    try:
        payload = None
        body_zone = raw_body.get("zone_id") if isinstance(raw_body, dict) and "zone_id" in raw_body else None
        if body_zone:
            zone_id = int(body_zone)

        # 1. Detectar payload explícito o usar últimas lecturas
        if isinstance(raw_body, dict) and "payload" in raw_body:
            payload = raw_body["payload"]
        else:
            candidate_keys = set(raw_body.keys())
            features_set = set(trainer.FEATURE_COLUMNS)
            if candidate_keys & features_set:
                payload = {k: float(v) for k, v in raw_body.items() if k in features_set}
            else:
                payload = None

        if payload is None or len(payload) == 0:
            if zone_id is None:
                raise HTTPException(status_code=400, detail="Falta zone_id o payload con lecturas.")
            latest = obtener_ultimas_lecturas_de_zona(zone_id)
            if not latest:
                raise HTTPException(status_code=400, detail="No hay lecturas para la zona.")
            payload = {k: float(v["valor"]) for k, v in latest.items()}

        if zone_id is None:
            raise HTTPException(status_code=400, detail="Falta zone_id.")

        # 2. Validar columnas necesarias
        missing = [c for c in trainer.FEATURE_COLUMNS if c not in payload]
        if missing:
            raise HTTPException(status_code=400, detail=f"Faltan columnas en payload: {missing}")

        # 3. Verificar si ya existe modelo entrenado
        zone_dir = os.path.join("outputs", f"zone_{zone_id}")
        model_path = os.path.join(zone_dir, f"modelo_qsvc_zone_{zone_id}.joblib")
        scaler_path = os.path.join(zone_dir, f"scaler_qsvc_zone_{zone_id}.joblib")

        need_retrain = False

        # ✅ Solo entrenar si NO existe modelo — nunca reentrenar automáticamente
        pca_path = os.path.join(zone_dir, f"pca_quantum_zone_{zone_id}.joblib")
        encoder_path = os.path.join(zone_dir, f"label_encoder_zone_{zone_id}.joblib")

        if not (os.path.exists(model_path) and os.path.exists(scaler_path)
                and os.path.exists(pca_path) and os.path.exists(encoder_path)):
            logging.info("Modelo no encontrado para zona %s → entrenando por primera vez...", zone_id)
            need_retrain = True
        # ✅ Si ya existe modelo, NUNCA reentrenar automáticamente desde /predict

        if need_retrain:
            logging.info("Entrenando zona %s por primera vez...", zone_id)
            trainer.train_zone(zone_id)
        # 6. Predecir con el modelo existente (ahora con probabilidades y comparación)
        clase_pred, X_scaled, zone_dir, probabilidades_qsvc, comparacion_clasica = \
            trainer.predict_from_values(zone_id, payload)
        
        sensibilidad_todos = trainer.analizar_sensibilidad_todos_cultivos(zone_id, payload)
        interpretacion = trainer.interpretacion_agronomica(payload)


        # Análisis de sensibilidad (qué mover para favorecer otro cultivo)
        try:
            sensibilidad = trainer.analizar_sensibilidad(zone_id, payload)
        except Exception:
            logging.exception("No se pudo calcular sensibilidad")
            sensibilidad = None

        interpretacion_simple = trainer.generar_interpretacion_simple(
            clase_pred, probabilidades_qsvc, comparacion_clasica, sensibilidad
        )


        try:
            evaluacion_lote = trainer.evaluar_lote_zona(zone_id)
        except Exception:
            logging.exception("No se pudo calcular evaluación por lote")
            evaluacion_lote = None

        base = f"/outputs/zone_{zone_id}/"
        imgs = {
            "pca": base + "superposicion_pca.png",
            "clusters": base + "clustering_emergente.png",
            "importance": base + "importancia_sensores.png",
            "stats": base + "estadisticas_entrenamiento.csv",
            "bloch": base + "bloch_superposicion.png"
        }

        # Tabla comparativa del paper, si ya se generó en el entrenamiento
        tabla_paper_path = os.path.join(zone_dir, "comparacion_modelos_paper.csv")
        tabla_paper = None
        if os.path.exists(tabla_paper_path):
            import pandas as pd
            tabla_paper = pd.read_csv(tabla_paper_path).to_dict(orient="records")

        return {
            "status": "ok",
            "zone_id": zone_id,
            "clase": str(clase_pred),
            "confianza_qsvc": probabilidades_qsvc.get(str(clase_pred)),
            "probabilidades_qsvc": probabilidades_qsvc,
            "comparacion_clasica": comparacion_clasica,

            "sensibilidad_todos_cultivos": sensibilidad_todos,   # <-- NUEVO
            "sensibilidad": sensibilidad,
            "tabla_comparativa_cv": tabla_paper,
            "interpretacion": interpretacion,
            "interpretacion_simple": interpretacion_simple,
            "imagenes": imgs,
            "input_used": payload
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Error en /predict")
        raise HTTPException(status_code=500, detail=str(e))
    




@app.get("/evaluar_lote")
def evaluar_lote(zone_id: int, n_muestras: Optional[int] = None):
    try:
        return trainer.evaluar_lote_zona(zone_id, n_muestras)
    except Exception as e:
        logging.exception("Error en /evaluar_lote")
        raise HTTPException(status_code=400, detail=str(e))
    





@app.get("/sensibilidad")
def sensibilidad(zone_id: int):
    try:
        latest = obtener_ultimas_lecturas_de_zona(zone_id)
        if not latest:
            raise HTTPException(status_code=400, detail="No hay lecturas para la zona.")
        payload = {k: float(v["valor"]) for k, v in latest.items()}
        return trainer.analizar_sensibilidad(zone_id, payload)
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Error en /sensibilidad")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def root():
    return {"status": "ok", "message": "Quantum Agriculture API - use /docs"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 9000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
