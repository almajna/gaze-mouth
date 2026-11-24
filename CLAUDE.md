@AGENTS.md

## Claude Code Workflow Rules
- Always write the plan to `docs/plan.md` before writing code.
- Use MediaPipe Tasks `FaceLandmarker` (`face_landmarker.task`), which outputs 478 landmarks plus 52 ARKit blendshapes (use blendshapes like `jawOpen` / `mouthPucker` alongside landmarks for mouth gestures).
