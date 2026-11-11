# Revisión visual del augmentation PyTorch

Fecha: 2026-11-11  
Ejecución rápida no canónica: Python 3.13.7, PyTorch 2.13.0+cpu, seed 42.

Evidencia: `reports/augmentation_preview_pytorch.png` y `reports/augmentation_validation.json`.

## Observaciones

- Se revisaron A, DEL, H, L, O, S, V y Z.
- Las ocho manos siguen siendo reconocibles después de augmentation.
- No se observó flip horizontal.
- Rotación, traslación, zoom, brillo y contraste son visibles y permanecen dentro de `[0,1]`.
- El relleno `reflection` genera franjas o triángulos reflejados en algunos bordes, más notorios en A, H y V. Es coherente con la política elegida y no oculta la mano, pero debe vigilarse en el sanity check del entorno canónico.
- Con `augmentation.eval()`, los cuatro ejemplos de validación y los cuatro de prueba comparados quedaron bit a bit idénticos a su entrada.

La inspección no demuestra equivalencia píxel a píxel con TensorFlow; confirma que la política preserva la semántica visual y que validación/prueba no reciben transformaciones aleatorias.
