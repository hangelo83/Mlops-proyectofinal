"""
api/schemas.py — Modelos Pydantic para validación de entrada y salida.
Pydantic valida automáticamente tipos y rangos.
Si falla -> HTTP 422 con mensaje claro, el modelo no recibe datos corruptos.
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field


class PacienteInput(BaseModel):
    """Datos crudos de un asegurado, tal como los registra la aseguradora."""
    Edad:                   float = Field(..., ge=0, le=120, description='Edad en años')
    Flag_hipertension:      int   = Field(..., ge=0, le=1, description='1 si tiene hipertensión')
    Flag_problem_cardiaco:  int   = Field(..., ge=0, le=1, description='1 si tiene problema cardiaco')
    Promedio_nivel_glucosa: float = Field(..., gt=0, le=400, description='Glucosa promedio (mg/dL)')
    IMC:                    Optional[float] = Field(None, ge=10, le=100,
                                                    description='Índice de masa corporal (opcional)')
    Estados_civil:          Literal['Si', 'No'] = Field(..., description='Si = alguna vez casado')
    Tipo_trabajo:           Literal['Emprendedor', 'Empresa_privada', 'En_gobierno',
                                    'Nunca_trabajo', 'cuidar_ninos']

    model_config = {'json_schema_extra': {'example': {
        'Edad':                   67,
        'Flag_hipertension':      1,
        'Flag_problem_cardiaco':  1,
        'Promedio_nivel_glucosa': 228.7,
        'IMC':                    36.6,
        'Estados_civil':          'Si',
        'Tipo_trabajo':           'Empresa_privada',
    }}}


class PrediccionOutput(BaseModel):
    """Respuesta de la API con el resultado de la predicción."""
    probabilidad_infarto: float                       # [0-1]
    decision:             Literal['PROGRAMA PREVENCION', 'SEGUIMIENTO', 'RUTINA']
    nivel_riesgo:         Literal['BAJO', 'MODERADO', 'ALTO', 'MUY ALTO']
    umbral_usado:         float                       # UMBRAL_PREVENCION
    modelo:               str                         # tipo de algoritmo usado


class HealthResponse(BaseModel):
    """Respuesta del endpoint de salud."""
    status:        str
    modelo:        str
    origen_modelo: str        # 'registry:Production' o 'pickle'
    version:       str
    recall_test:   float
    auc_test:      float
    env:           str
