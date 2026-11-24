# Architecture

```text
 Webcam frames                               Echo-processed microphone PCM
   |                                            |
   +--> face landmarks / jawOpen                +--> audio speech edges
   |      |                                                |
   |      +--> visual speech edges ------+                 |
   |      +--> jaw-click candidates -----+                 |
   |                                     v                 v
   |                              +-------------------------------+
   |                              | ModeArbitrator                |
   |                              | ACTIVE  gaze + visual + mic   |
   |                              | STEALTH gaze + visual         |
   |                              | SLEEP   pause control         |
   |                              | wake -> last active mode      |
   |                              +-------------------------------+
   |                                |           |          |
   |                       accepted click    visual      ACTIVE only
   |                                |           |          |
   +--> GazeBackend                 |      mouth crops  Whisper PCM
          |                         |           |          |
          v                         v        Auto-AVSR    Whisper <--- AX hotwords
       gaze point ----------> snap / dispatch   |          |
          |                         |           +----+-----+
          |          verified click target           |
          |                         |              command
          v                         v                |
       GazeLoupe geometry   PassiveClickCalibrator   |
          ^                         |                |
          +---- corrected gaze <---+                v
                                           +---------------------+
                                           | AXUIElement context |
                                           +----------+----------+
                                                      |
                                      AX text --------+-----+ pixel fallback
                                      |                     v
                                      |               FastVLM-0.5B
                                      v                     |
                                    Qwen3-4B <--------------+
                                      |
                                      v
                            action / HUD -> observe UI again
```

## Component boundaries

- `handsfree/audio_arbitration.py` defines the three-mode policy. Speech that begins with a narrow jaw can block clicks through a release cooldown. An accepted click starts a separate visual-speech cooldown. This module consumes detector edges and click candidates from its caller.
- `handsfree/gaze_engine.py` defines `GazeBackend`, the GazeFollower adapter, and Gaze Loupe coordinate transforms. The passive calibrator updates a local correction grid from settled physical clicks or jaw clicks snapped to a verified AX target. The caller supplies those events. `BlazeGazeBackend` delegates to GazeFollower while the WebEyeTrack adapter is unavailable.
- `handsfree/mac_ui_bridge.py` traverses the focused AX window with depth and node caps. `ContextHotwordBiaser` extracts a short list of named controls for Whisper's `initial_prompt`.
- `handsfree/agent_planner.py` reads AX context at the gaze point. When it needs visual context, it sends a gaze-centered screenshot crop to FastVLM-0.5B. Qwen3-4B selects a tool or a HUD response. The planner checks for a UI change between actions and caps the loop at four steps.

The live controller applies these policies in ACTIVE, STEALTH, and SLEEP. ACTIVE accepts microphone and visual input. STEALTH stops the microphone stream and recognizes mouthed commands. SLEEP pauses action dispatch and preserves the gesture or keyboard wake path.

Pursuit calibration fits ridge regression followed by a thin-plate spline correction and reuses compatible saved anchors. The HUD reticle smooths its displayed position independently of the click target. The local MCP client discovers configured stdio tools and makes their schemas available to the planner.
