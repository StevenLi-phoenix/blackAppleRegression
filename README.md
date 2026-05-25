# Bad Apple!! — Recursive Regression Renderer

Render the classic [Bad Apple!!](https://www.youtube.com/watch?v=FtutLA63Cp8) video using recursive pattern matching and frame-by-frame regression placement.

Each frame is binarized, a dominant pattern is extracted, then recursively placed at decreasing scales to approximate the original — resulting in a fractal-like mosaic effect.

> **⚠️ Photosensitivity Warning**: This video contains rapid flashing patterns and high-contrast flickering. Viewer discretion is advised for those with photosensitive epilepsy.

## Demo

[![YouTube](https://img.shields.io/badge/YouTube-Watch-FF0000?logo=youtube)](https://youtu.be/IF1Y2DUaxM8)
[![Bilibili](https://img.shields.io/badge/Bilibili-Watch-00A1D6?logo=bilibili)](https://www.bilibili.com/video/BV1Bp2YBzEMR/)

## How it works

1. **Binarize** each frame (adaptive threshold)
2. **Extract** the dominant connected pattern from the binary image
3. **Recursive placement**: from large to small (`MINIMAL_PATTERN_SIZE=10`), find open spaces in the frame and stamp the pattern at decreasing sizes
4. **Caching**: frame hashes → disk cache to skip redundant computation
5. **Multiprocessing**: parallel frame processing with configurable worker count
6. **Audio merge**: final output is merged with the original audio track via ffmpeg

## Usage

```bash
# Place BadApple.mp4 in the project root
pip install -r requirements.txt
python main.py
# Output: out.mp4 (video only), merged.mp4 (with audio)
```

## Requirements

- Python 3.12+
- OpenCV, NumPy, Numba, tqdm
- ffmpeg (for audio merge)

## License

MIT
