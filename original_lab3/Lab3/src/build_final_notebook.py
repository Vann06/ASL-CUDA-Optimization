"""Construye y ejecuta el notebook final a partir del informe consolidado."""

from __future__ import annotations

import re
from pathlib import Path

import nbformat
from nbclient import NotebookClient


LAB_DIR = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = LAB_DIR / "Notebooks"
REPORT_PATH = LAB_DIR / "INFORME_FINAL_LAB3.md"
OUTPUT_PATH = NOTEBOOKS_DIR / "02_random_forest_comparacion_final.ipynb"


CODE_AFTER_SECTION = {
    "## 3.": """import pandas as pd
from pathlib import Path
from IPython.display import display, Image

RESULTS_DIR = Path('results')
tabla_dl = pd.read_csv(RESULTS_DIR / 'tabla_final.csv')
display(tabla_dl.style.format({
    'acc_val': '{:.2%}', 'acc_test': '{:.2%}', 'f1_macro': '{:.2%}',
    'tiempo_entreno_s': '{:.1f}'
}))""",
    "## 4.": """candidatos_rf = pd.read_csv(RESULTS_DIR / 'rf_candidates.csv')
display(candidatos_rf.style.format({
    'accuracy_train': '{:.2%}', 'accuracy_val': '{:.2%}',
    'f1_macro_val': '{:.2%}', 'tiempo_entreno_s': '{:.2f}'
}))""",
    "## 5.": """comparacion = pd.read_csv(RESULTS_DIR / 'comparacion_principal.csv')
display(comparacion.style.format({
    'accuracy_test_recalculada': '{:.2%}',
    'f1_macro_test_recalculado': '{:.2%}'
}))
display(Image(filename=str(RESULTS_DIR / 'confusion_matrix_rf.png')))""",
    "## 6.": """comparacion_ae = pd.read_csv(RESULTS_DIR / 'comparacion_A_E_interna.csv')
predicciones_ae = pd.read_csv(RESULTS_DIR / 'predicciones_oficiales_A_E.csv')
display(comparacion_ae.style.format({'accuracy_A_E': '{:.2%}', 'f1_macro_A_E': '{:.2%}'}))
display(predicciones_ae)""",
    "## 7.": """resumen_propias = pd.read_csv(RESULTS_DIR / 'comparacion_fotos_propias.csv')
por_letra = pd.read_csv(RESULTS_DIR / 'fotos_propias_por_letra.csv')
por_integrante = pd.read_csv(RESULTS_DIR / 'comparacion_fotos_propias_por_integrante.csv')
display(resumen_propias.style.format({
    'accuracy_fotos_propias': '{:.2%}', 'confianza_media': '{:.2%}'
}))
display(por_integrante.style.format({'accuracy': '{:.2%}', 'confianza_media': '{:.2%}'}))
display(por_letra.pivot(index='letra', columns='modelo', values='recall').style.format('{:.0%}'))

import matplotlib.pyplot as plt
aporte = por_integrante[por_integrante['modelo'] == 'CNN2_aug']
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(aporte['integrante'], aporte['n_fotos'], color='#4C78A8')
ax.set(title='Aporte equilibrado de fotos propias', ylabel='Número de fotos', ylim=(0, 60))
for bar, letras in zip(bars, aporte['n_letras']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{int(bar.get_height())} fotos / {int(letras)} letras', ha='center')
plt.tight_layout()
plt.show()

display(Image(filename=str(RESULTS_DIR / 'ejemplos_fotos_propias.png')))
display(Image(filename=str(RESULTS_DIR / 'confusion_fotos_propias_cnn_aug.png')))""",
}


def build_notebook() -> nbformat.NotebookNode:
    report = REPORT_PATH.read_text(encoding="utf-8")
    sections = re.split(r"(?=^## )", report, flags=re.MULTILINE)
    cells = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        cells.append(nbformat.v4.new_markdown_cell(section))
        for prefix, code in CODE_AFTER_SECTION.items():
            if section.startswith(prefix):
                cells.append(nbformat.v4.new_code_cell(code))
                break

    cells.append(
        nbformat.v4.new_markdown_cell(
            "## Reproducción\n\n"
            "Desde la raíz del repositorio, ejecute:\n\n"
            "```powershell\n"
            ".\\.venv\\Scripts\\python.exe Lab3\\src\\prepare_asl_data.py\n"
            ".\\.venv\\Scripts\\python.exe Lab3\\src\\random_forest_pipeline.py\n"
            "```\n\n"
            "La segunda orden reutiliza características y modelo si ya existen. "
            "Para forzar un entrenamiento nuevo, elimine manualmente los artefactos "
            "generados de `Notebooks/data` y `Notebooks/models` después de confirmar "
            "que no necesita conservarlos."
        )
    )
    cells.append(
        nbformat.v4.new_code_cell(
            """archivos_clave = [
    Path('mejor_modelo.keras'),
    Path('mejor_modelo_aug.keras'),
    Path('models/random_forest_hog_color.joblib'),
    RESULTS_DIR / 'metrics_random_forest.json',
    RESULTS_DIR / 'predicciones_fotos_propias.csv',
]
pd.DataFrame({
    'archivo': [str(path) for path in archivos_clave],
    'existe': [path.exists() for path in archivos_clave],
    'tamano_MB': [round(path.stat().st_size / 1024**2, 2) if path.exists() else None for path in archivos_clave],
})"""
        )
    )

    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata.language_info = {"name": "python", "version": "3.10"}
    return notebook


def main() -> None:
    notebook = build_notebook()
    client = NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOKS_DIR)}},
    )
    client.execute()
    nbformat.write(notebook, OUTPUT_PATH)
    print("Notebook final creado y ejecutado:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
