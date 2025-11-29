"""Entrena y guarda el preprocesador ACV (imputación + z-score + PCA).

Este script debe ejecutarse una vez que se tenga disponible el dataset
`nhanes_stroke_clean.csv` generado por el equipo de Data Science.

Resultado:
    - models/preprocessor_acv.pkl
"""

from pathlib import Path

import pandas as pd

from core.config_features import MODEL_INPUT_COLUMNS
from core.preprocessing import fit_preprocessor


def main() -> None:
    """Entrena el preprocesador sobre `nhanes_stroke_clean.csv`."""
    project_root = Path(__file__).resolve().parents[2]

    data_path = (
        project_root
        / "ml_models"
        / "data"
        / "processed"
        / "nhanes_stroke_clean.csv"
    )
    output_path = project_root / "models" / "preprocessor_acv.pkl"

    if not data_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de datos esperado:\n{data_path}"
        )

    print(f"📄 Leyendo datos desde: {data_path}")
    df = pd.read_csv(data_path)

    # Tomar solo las columnas de entrada definidas en el contrato
    missing_cols = [c for c in MODEL_INPUT_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Faltan columnas en el dataset para entrenar el preprocesador: {missing_cols}"
        )

    X = df[MODEL_INPUT_COLUMNS]

    print("🔧 Ajustando preprocesador (imputación + z-score + PCA)...")
    fit_preprocessor(X, save_path=output_path)

    print(f"✅ Preprocesador guardado en: {output_path}")


if __name__ == "__main__":
    main()


