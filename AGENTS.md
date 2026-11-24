# Shared Project Rules
1. NO MOCK CAMERA OR SYNTHETIC TESTS: Never mock `cv2.VideoCapture`, MediaPipe outputs, or generate fake/random landmark arrays.
2. REAL DATA ONLY: All metrics, thresholds, and accuracy claims must come from running scripts against real files in `data/`.
3. KEEP IT FLAT: Single-purpose scripts (`capture.py`, `eval_offline.py`, `live_app.py`). No plugin architectures, async buses, or premature abstractions.
4. FAILURE MODES FIRST: Always report false positives on `negative` clips and cross-session error rates before overall accuracy.
5. HARDWARE BOUNDARY: When a script needs a live webcam or human trials, stop and output the exact command for the user to run.
