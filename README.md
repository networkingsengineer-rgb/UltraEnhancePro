# UltraEnhance Pro 1.0

Free AI photo and video upscaler for Windows 10 and 11 (64-bit).
Upscale and enhance to **720p, 1080p, 2K, 4K and 8K**. No watermarks, no
subscriptions, no account. Everything (Python, every package, FFmpeg and the
AI models) is inside the installer: you do not install Python and you never
need the Command Prompt.

---

## 1. Install (for everyone)

**Installer**
1. Double-click `UltraEnhancePro_Setup.exe`.
2. Choose *Install for me only* (no admin password needed) or *Install for all users*
   (goes to Program Files and asks for admin).
3. Tick *Create a desktop shortcut* if you want one, press **Install**, then **Finish**.
4. Start it from the Start menu or the desktop icon.
   To remove it: Settings > Apps > UltraEnhance Pro > Uninstall.

**Portable (no install)**
1. Right-click `UltraEnhancePro_Portable.zip` > *Extract All*.
2. Open the extracted folder and double-click `UltraEnhancePro.exe`.
   You can run it from a USB stick.

If Windows SmartScreen says "Windows protected your PC", click *More info* >
*Run anyway* (the app is not code-signed because signing certificates cost money).

## 2. System requirements

| | Minimum | Recommended |
|---|---|---|
| Windows | 10 64-bit (1809 or newer) | 11 64-bit |
| Memory | 8 GB RAM | 16 GB RAM or more |
| Graphics | Any Vulkan GPU with 2 GB VRAM (NVIDIA, AMD or Intel) | 6 GB VRAM or more |
| Disk | 2 GB for the app + space for temporary frames | SSD with 100 GB+ free for 4K/8K video |

No GPU? Choose *CPU only* in Settings: it works, but is many times slower.

## 3. How to use

### Photos
1. **Photo Enhance** tab: drag in photos or whole folders, or click the drop area.
   Supported: JPG, JPEG, PNG, WEBP, BMP, TIFF, HEIC/HEIF, GIF (first frame), TGA, PPM.
2. Pick a **preset** (720p, 1080p, 2K, 4K, 8K) or a **scale** (2x, 3x, 4x, 8x; 8x runs two AI passes).
3. **Aspect ratio**: *Fit* keeps the whole image, *Fill* fills the frame and crops edges, *Pad* adds black bars.
4. **AI model**: *Photo* (real photos), *Anime / Art* (drawings), *Old photo restore*
   (switches on face restore, denoise, colour fix and JPEG clean-up).
5. Optional: Face restore (GFPGAN), Denoise, Sharpen, Auto-fix colour/brightness/contrast, Remove JPEG artefacts.
6. **Preview** shows a before/after of the centre of the photo: drag the white line, mouse-wheel to zoom,
   double-click to fit.
7. Choose JPG (quality slider), PNG, WEBP or TIFF. *Keep EXIF* keeps camera, date and GPS data.
8. Press **Enhance photos**. Files are saved as `name_4K_enhanced.jpg` etc. Originals are never changed.

### Videos
1. **Video Enhance** tab: drag in videos (MP4, M4V, MKV, MOV, AVI, WEBM, WMV, FLV, MPG, TS, M2TS, 3GP, OGV, VOB).
2. Set the **export settings**: resolution, aspect ratio, AI model, smooth motion (24/30 to 60 fps),
   deinterlace, denoise, stabilise, codec (H.264, H.265/HEVC, AV1), encoder (NVIDIA NVENC, AMD AMF,
   Intel Quick Sync or software), quality (Fast/Balanced/Best), CRF or bitrate, container (MP4/MKV/MOV/WEBM).
3. Select one or more videos (Ctrl+click / Shift+click) and press **Apply settings to selected**.
   Each video keeps its own settings, shown in the *Export settings* column. *Apply settings to all* does every video.
4. **Estimate** shows output size, temporary disk space and an honest time estimate.
5. **Preview 5 seconds** processes a 5-second sample from the middle and opens it.
6. Press **Enhance videos**. Original audio and subtitles are copied and stay in sync.

### Batch Queue
Every job shows progress and time left. Pause, resume, cancel, retry, reorder, remove finished.
Double-click a finished job to open the result. If the PC turns off during a video, open the app again:
the **Home** tab lists unfinished jobs and resumes from the last finished chunk of frames.

### Settings
GPU (auto, CPU or a specific card), tile size (0 = automatic from VRAM), threads, temp folder,
output folder, default preset, language (English / Hindi), dark or light theme.

## 4. Troubleshooting

| Problem | Fix |
|---|---|
| "Your graphics card ran out of memory" | Settings > Tile size: 128, or 64 on 2 GB cards. Press *Retry* in the Batch Queue. |
| "No compatible graphics driver was found (Vulkan)" | Update the GPU driver from nvidia.com, amd.com or intel.com and restart. Or choose *CPU only*. |
| Output is black or has green blocks | Old Intel iGPU driver: update it, or select your other GPU in Settings. |
| Hardware encoder not used | It is only used if a real test encode works. Update the driver; the app falls back to x264/x265 automatically. |
| "Not enough free disk space" | Free space or pick a temp folder on a bigger drive (Settings > Temp folder). 4K/8K video needs a lot. |
| Very slow | Normal for 4K/8K. 8K can take many hours per minute of video. Use *Fast* quality, a lower preset, or the animevideov3 model. |
| HEIC photo will not open | Re-save it from the Photos app as JPG, or send the log file below. |
| App will not start | Delete `%USERPROFILE%\UltraEnhancePro\settings.json` and try again. |
| Anything else | Tools > Open log file. The log is at `%USERPROFILE%\UltraEnhancePro\log.txt`. |

---

## 5. For developers: building the installer

Users never do this. It is for whoever produces `UltraEnhancePro_Setup.exe`.

**Option A: GitHub (no Windows PC setup needed)**
Push this folder to a GitHub repository, open *Actions* > *Build Windows installer* > *Run workflow*.
After about 15 minutes download the `UltraEnhancePro-windows` artifact: it holds
`UltraEnhancePro_Setup.exe` and `UltraEnhancePro_Portable.zip`.

**Option B: on a Windows PC**
1. Install Python 3.11 (64-bit) from python.org and Inno Setup 6 from jrsoftware.org (both free).
2. Double-click `build.bat`. It creates a virtual environment, installs the packages, downloads FFmpeg,
   Real-ESRGAN, RIFE and the face models into `vendor\`, runs the tests, builds with PyInstaller and
   creates both the installer and the portable ZIP in `dist\`.

Run the tests alone with `run_tests.bat`. Run from source with `python run_app.py`
(after `python tools\fetch_deps.py` once).

### Project layout
```
UltraEnhancePro/
  run_app.py                 entry point
  ultraenhance/
    main.py                  start-up, logging, crash dialog
    config.py                settings, paths, log file
    formats.py               supported file types, folder scanning
    presets.py               720p..8K presets, fit/fill/pad maths, AI pass planning
    tiling.py                VRAM -> safe tile size
    gpu.py                   GPU/VRAM detection, CPU/GPU load
    engines.py               bundled tools + Real-ESRGAN / RIFE command builders
    ffmpeg_utils.py          probe, encoder detection, FFmpeg command builders
    runner.py                child-process runner with progress/pause/cancel
    job_queue.py             batch queue (pause/resume/cancel/retry)
    estimates.py             disk space and time estimates
    photo.py                 photo pipeline
    face_restore.py          GFPGAN (ONNX) + YuNet face restore
    video.py                 resumable chunked video pipeline
    ui/                      PySide6 window, widgets, themes, translations
  tests/                     unit tests (pytest)
  tools/fetch_deps.py        downloads engines/models for the build
  tools/make_icon.py         draws the app icon
  installer/UltraEnhancePro.iss
  UltraEnhancePro.spec       PyInstaller recipe
  build.bat  run_tests.bat  requirements.txt  requirements-dev.txt
  licenses/THIRD_PARTY_NOTICES.txt
  .github/workflows/build-windows.yml
```

### How the video engine works
1. FFmpeg decodes to PNG frames at a constant frame rate (optional vidstab two-pass, bwdif deinterlace, hqdn3d denoise).
2. Optional RIFE interpolation to the target fps on the source-size frames (cheaper, same duration, so audio stays in sync).
3. Frames are upscaled by Real-ESRGAN in chunks with a VRAM-safe tile size; each chunk is encoded straight away
   into a segment and its PNGs are deleted, so 4K/8K never needs all frames on disk at once.
4. Segments are joined without re-encoding, then audio, text subtitles, chapters and metadata are copied from the source.
5. A `state.json` per job records finished chunks; a restarted job skips them.
