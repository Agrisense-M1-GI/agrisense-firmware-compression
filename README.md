# algo/mjpeg-webcam

Compression MJPEG embarquée dans la webcam USB (référence la plus simple).

## Principe

La webcam UVC compresse elle-même les images en MJPEG via son DSP interne.
Le Pi récupère directement le flux compressé via ffmpeg sans aucun traitement supplémentaire.

## Limites

- PSNR et SSIM non calculables (pas d'accès à l'image brute avant compression)
- Qualité fixée par le driver webcam

## Utilisation

```bash
python3 pipeline.py
```
