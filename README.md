# AgriSense - Firmware Nœud Capteur

Pipeline de compression d'images pour WSN agricole.

## Branches

| Branche | Description |
|---|---|
| `algo/mjpeg-webcam` | Compression MJPEG embarquée webcam USB (référence) |
| `algo/jpeg-baseline` | JPEG standard libjpeg IJG v9e |
| `algo/agriJPEG-v1` | AgriJPEG v1 (Q_vegetation + Q_fond + 4:1:4 + rANS) |
| `algo/ADRES` | ADRES |
| `algo/OTS-WZ` | OTS-WZ |

## Utilisation

```bash
python3 pipeline.py
```
