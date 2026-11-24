# Detailed setup and usage guide

This guide preserves setup details and recorded results from the original README. The assistant has received changes since some of its live-check instructions were written; see the current source and evaluation report when checking behavior.

An Apple Silicon macOS prototype for eye-gaze pointing, jaw clicks, hand gestures, and commands issued by mouthing or whispering. It has two entry points.

| Entry point | Use it for |
| --- | --- |
| `handsfree_mac_agent.py` | Desktop control. Calibrated GazeFollower gaze, focused-window Accessibility targets, gestures, a small silent-command classifier, optional open-vocabulary lip reading, and Whisper. |
| `live_lipread.py` | Sentence dictation. Space starts and stops recording in the preview. The 25 Hz mouth crops feed USR 2.0 with the speaker LoRA and optional Qwen correction. Text appears in the preview and terminal. Paste requires `--paste`. |

See [ARCHITECTURE.md](../ARCHITECTURE.md) for the implementation and [docs/plan.md](../docs/plan.md) for experiment methods and results.

## Set up on Apple Silicon

Install Xcode Command Line Tools and Homebrew, then run the following from the checkout. The lock was verified on arm64 macOS with CPython 3.12.16. KenLM builds from source and needs a compiler and CMake.

```sh
xcode-select --install       # once, if Command Line Tools are missing
brew install uv cmake
uv venv --python 3.12 .venv-gf
uv pip sync --python .venv-gf/bin/python --require-hashes requirements.lock
uv pip check --python .venv-gf/bin/python
.venv-gf/bin/python -c 'import cv2; cv2.face.createFacemarkLBF()'
.venv-gf/bin/python setup_assets.py
.venv-gf/bin/python setup_assets.py --check
```

The lock uses NumPy 2 and a hash-pinned archive of the pending pyctcdecode compatibility patch. GazeFollower and MLX-VLM require `opencv-python`, and the landmark cropper uses `cv2.face` from contrib. Both distributions match the recorder's OpenCV 5.1.0 version. If the landmark import check fails, reinstall the contrib wheel last with `uv pip install --python .venv-gf/bin/python --reinstall-package opencv-contrib-python --no-deps --require-hashes -r requirements.lock`. [pyctcdecode compatibility patch](https://github.com/kensho-technologies/pyctcdecode/pull/118)

Use `setup_assets.py --planner` for Qwen3 and FastVLM. The desktop agent requires the command net, a validated trigger config, and the original fixation recordings or equivalent recordings made through `training/capture.py` and evaluated with `eval/eval_offline.py`. Run `eval/eval_offline.py` in a fresh workspace or preserve the existing calibration first. It replaces `logs/trigger_config.json` on pass and can remove it on failure. Sentence dictation uses the speaker LoRA under `models/usr2_speaker_lora_v2/` by default. Use the locally trained copy or train a new adapter into a fresh directory and pass `--adapter`. Obtain the recordings and trained local weights separately with authorization. [Asset inventory](../ASSETS.md)

## Replay the recorded checks

These commands read recorded data and local weights. They do not use camera, microphone, or live desktop input.

```sh
.venv-gf/bin/python -m handsfree.gaze_engine
.venv-gf/bin/python -m eval.verify_audio_locks
.venv-gf/bin/python -m eval.eval_gaze_scroll --compare-scroll session_1 session_3
.venv-gf/bin/python -m unittest tests.test_usr_tokenizer tests.test_sentence_blitz \
  tests.test_context_decoder.WordSearchTests tests.test_context_decoder.EndToEndTests \
  tests.test_handsfree_modes.OpenVocabRoutingTests \
  tests.test_handsfree_modes.ModeTests.test_active_segment_uses_audio_from_the_recorded_preroll \
  tests.test_handsfree_modes.ModeTests.test_gaze_paced_scroll_and_click_drift_on_recorded_clips
.venv-gf/bin/python -m eval.run_benchmarks --json
.venv-gf/bin/python -m eval.eval_wer --backend usr2 --decode joint --device mps
.venv-gf/bin/python -m eval.eval_wer --backend usr2 --decode joint --device mps \
  --adapter models/usr2_speaker_lora_v2
.venv-gf/bin/python -m eval.eval_wer --backend usr2 --decode joint --device mps \
  --adapter models/usr2_speaker_lora_v2 --data data/nonspeech_25hz --min-ctc-tokens 5
.venv-gf/bin/python live_lipread.py --device mps --replay data/eval_holdout/001.npz
```

The replay suite needs the original `data/session_1` through `data/session_3` MP4/NPZ pairs and, for `eval/run_benchmarks.py`, `logs/trigger_config.json`, the original speech sessions, and a take under `data/silent_train/`. The sentence dataset currently has 50 holdout and 260 training archives. Inspect `python -m training.record_sentences --help` for data collection. Clip counts and scores are specific to the local recordings.

## Recorded results

Scroll settings were explored on sessions 1 and 2. The comparison fits calibration on session 1 and replays session 3. Pixel totals are the absolute amount scrolled in either direction for negatives, or the intended direction for fixation clips.

| Session 3 replay | Previous clips / pixels | Current clips / pixels |
| --- | ---: | ---: |
| Center false scroll | 3/6 · 200 px | 3/6 · 40 px |
| Look-down false scroll | 3/3 · 1,500 px | 2/3 · 455 px |
| Intentional scroll down | 6/6 · 7,351 px | 6/6 · 6,215 px |
| Intentional scroll up | 1/6 · 20 px | 1/6 · 20 px |

The median time to the first intended downward step increased from 244 ms to 407 ms. Wrong-direction steps on top fixations fell from 2/6 clips to 0/6. The low upward recall and residual false scroll need live validation.

| Measurement | Recorded result | Source and scope |
| --- | --- | --- |
| Jaw-click false positives | 0/27 | Saved trigger on blink, swallow and look-down clips, `python -m eval.run_benchmarks --json`. |
| Acoustic false locks on silent speech | 0/25 | Leave-one-session-out; whisper recall 11/20. `eval/verify_audio_locks.py`. |
| USR 2.0 speaker v2 text on non-speech | 63/64 ungated, 0/64 gated | `eval/eval_wer.py` joint decoding on `data/nonspeech_25hz` with the v2 adapter, A/B using `--min-ctc-tokens 5`. These clips were used to choose the gate. |
| Cross-session gaze quadrant error | 19.06–30.70% per frame; 16.67–30.77% per clip | Six calibration/test directions across three sessions, from `python -m eval.run_benchmarks --json`. The metric is quadrant classification error. |
| Sentence WER, USR 2.0 Base+ joint | 89.78% (334/372) | `python -m eval.eval_wer --backend usr2 --decode joint --device mps`. |
| Sentence WER, USR 2.0 speaker v2 joint | 62.37% (232/372) | Same command with `--adapter models/usr2_speaker_lora_v2`. |
| Open-vocabulary decoder integration | The real test take “open terminal” produced “slower young” | `tests.test_context_decoder.EndToEndTests`. The test checks inference and output structure. |

The WER comparisons cover 50 clips from one recording session and one speaker. Cross-session sentence WER remains unmeasured. A `eval/run_benchmarks.py` replay measured CPU Auto-AVSR at 665.14 ms median and 882.98 ms p95 over 20 takes. The gaze stage measured 3.22 ms median and 3.38 ms p95 over 79 frames for Face Landmarker, iris/head features, and quadrant prediction. It excludes MGazeNet, video reads, and display updates. The final lock reproduced the stored crops exactly on `001.npz` in `live_lipread.py --replay`. That single clip took 288 ms after the simulated stop and produced an incorrect transcript. Live end-to-end latency remains unmeasured.

## Run sentence dictation

Grant **Camera** access to the launching terminal in **System Settings → Privacy & Security**. **Accessibility** access enables focused-element context and is needed for the optional paste workflow. Start in text-preview mode.

```sh
.venv-gf/bin/python live_lipread.py --device mps
```

Use **Space** in the preview to start and stop, **Q** to quit, and `--ger off` to inspect the visual-only transcript. To collect fresh labeled data, create `train_sentences.txt` with one prompt per line and run `.venv-gf/bin/python -m training.record_sentences --list train_sentences.txt`. Use a separate `eval_holdout_sentences.txt` list and recording session for evaluation. These files are local and ignored by Git. Context correction training also takes a local `--domains` JSON file with a one-based `index` and a `domain` for each training prompt. See `training/build_ger_data.py` for supported domains. The recorder's preview shows the mouth crop. Press **R** to redo the last clip.

## Run the live macOS agent

In **System Settings → Privacy & Security**, allow the launching terminal under **Accessibility**, **Input Monitoring**, and **Camera**. Add **Microphone** for Whisper and **Screen & System Audio Recording** for visual queries and the gaze loupe. [Apple Accessibility setup](https://support.apple.com/guide/mac-help/allow-accessibility-apps-to-access-your-mac-mh43185/mac)

```sh
.venv-gf/bin/python handsfree_mac_agent.py --dry-run --verify-interactions --silent-only
.venv-gf/bin/python handsfree_mac_agent.py --onboarding --silent-only
```

The first command replays recorded clips. The second starts live onboarding, which checks for your eyes, then guides you through nine dots. Hold steady on each dot for 0.8 seconds of stable tracking. The ring resets when your gaze moves. At the end, look around to check the blue dot during the three-second preview. Press **R** to repeat calibration or **Esc** to quit. The app saves eye calibration in `logs/trigger_config.json`. Subsequent launches without `--onboarding` load that configuration and skip the overlay.

| Control | Startup flag | Global shortcut | Command |
| --- | --- | --- | --- |
| Gaze tracking | `--no-gaze` starts it off | Ctrl+Option+G | “toggle gaze” |
| Gaze-paced scrolling | `--auto-scroll` | Ctrl+Option+S | “toggle scroll” |
| Silent-only | `--silent-only` | Ctrl+Option+M | “mute mic”, “silent mode”, or “unmute mic” |

The HUD shows gaze and microphone status. The gaze dot is hidden during browsing. A trigger briefly outlines the selected control. Use `--debug-gaze` to show the raw gaze point. With auto-scroll on, rest your gaze in the bottom quarter of the focused window for 380 ms to scroll down or at the top edge for 220 ms to scroll up. Downward scrolling stops when recent iris observations leave the range sampled at eye calibration. Looking back at the middle, blinking, speaking, or starting a hand action stops scrolling. `--dry-run --auto-scroll --debug-gaze` replays recorded clips and prints every scroll event.

Raise one open palm into the upper center of the camera view and hold it for a quarter second. The **Ask / Action** prompt freezes the gaze target or uses the active window with gaze off. Whisper or mouth a command, or click the prompt field, type your request, and press **Return**. **Esc** closes the prompt.

Silent-only stops microphone capture and uses visual speech plus jaw clicks. With gaze off, jaw clicks use the pointer in the focused window. When the microphone is on, the agent uses Core Audio echo cancellation when available. An incompatible route or an explicit `--mic` device selects sounddevice. The HUD shows detected voice or lip activity.

**Ctrl+Option+Space** pauses or resumes. **Calibrate Head/Hands** in the menu bar sets up the head ray and the additional pinch, flick, and spread controls. Those controls can hold a target, scroll, click, or open the radial ring. Say “zoom here” to open the loupe and press or say an available number from 1 to 9. Pass `--move-cursor` to warp the cursor when a click fires or `--no-sound` to disable sound cues. Use **Quit** in the menu bar or **Ctrl+C** in the terminal to exit.

### Visual queries and local tools

Look at the text or error and say “read this,” “what's this error,” or “summarize this.” The planner captures the area around gaze, sends a 512 × 512 RGB image to FastVLM, and displays the answer in the HUD. Provide Screen Recording permission to the launching terminal. Use `setup_assets.py --planner` to install both planner checkpoints before an offline demo.

The app remembers the three most recent fixated regions. After switching apps, say “summarize what I was just looking at” or “copy that error.” Focus a regular text field and say “paste it here” to paste the copied result.

To connect a local MCP server, create `mcp_servers.json` at the repository root. Set the executable and directory to paths on your Mac. The server runs with your local account's permissions. Use `allowedTools` to choose which discovered operations the planner can call.

```json
{
  "mcpServers": {
    "notes": {
      "command": "/absolute/path/to/mcp-server-filesystem",
      "args": ["/absolute/path/to/notes"],
      "allowedTools": ["list_directory", "read_text_file"]
    }
  }
}
```

## 60-second DeepMind Hackathon demo

Before the timer, calibrate and rehearse on the demo display. Open a Safari page with a large, harmless control. Use the same speaker and silent-command checkpoint from rehearsal. For the live run, start the calibrated app with `.venv-gf/bin/python handsfree_mac_agent.py`.

| Time | What to show | What to say |
| --- | --- | --- |
| 00–10 s | Show the page and ACTIVE HUD. Keep both hands in view. | “The webcam estimates where I'm looking. A jaw gesture commits the click.” |
| 10–22 s | Look at the prepared control, hold gaze, then drop your jaw once. Let the audience see the target respond. | “I look, then click.” |
| 22–36 s | Cycle once into STEALTH. Mouth “open terminal” and show the result. | After it runs, say “That command came from lip movement. The mic stream was stopped.” |
| 36–50 s | Cycle back to ACTIVE. Whisper “open Safari” and show Safari return to the front. | “Active mode routes this whispered command through Whisper.” |
| 50–60 s | Press Ctrl+Option+Space and show the paused HUD. | “Pause stops control. The same shortcut restores the previous mode.” |

## License

The repository code and its documentation use the [MIT License](../LICENSE). External dependencies, model weights, calibration files, and recordings retain their own terms.

- **GazeFollower and its bundled MGazeNet weights.** Upstream declares CC BY-NC-SA 4.0. Check the terms for the specific checkpoint you obtain. [GazeFollower license information](https://github.com/GanchengZhu/GazeFollower#license-information)
- **Auto-AVSR.** Upstream code uses Apache 2.0. The pretrained checkpoints can carry separate conditions derived from training datasets. Review those terms for the encoder and any artifacts you distribute with it. [Auto-AVSR license information](https://github.com/mpc001/auto_avsr#license)
- **Other models.** Consult the model cards for the weights you install, including Whisper, the live Qwen2.5 model, and any standalone planner models. [Whisper Turbo](https://huggingface.co/mlx-community/whisper-large-v3-turbo) · [Qwen2.5-3B](https://huggingface.co/mlx-community/Qwen2.5-3B-Instruct-4bit)

## CursorMix for macOS

CursorMix is a standalone menu-bar assistant. Press **Control–Option–Space (⌃⌥Space)** to capture the current display and focused window, type a question, and press Return. Escape dismisses the panel. **Settings → Keyboard Shortcut** changes the exclusive global shortcut.

### Setup

From the checkout on macOS with Python 3.12 and `uv` installed:

```sh
uv venv --python 3.12 .venv-cursormix
uv pip sync --python .venv-cursormix/bin/python --require-hashes cursormix/requirements.lock
.venv-cursormix/bin/python -m cursormix
```

Set the provider key in the checkout’s gitignored `.env.local` or in the launching shell. CursorMix and its evaluators load `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` from that file. Existing environment values take precedence. **Automatic** uses Claude first when `ANTHROPIC_API_KEY` is set, followed by OpenAI when `OPENAI_API_KEY` is set. With neither key it uses local inference. Select an explicit provider in Settings to override that choice. The first-run walkthrough accepts keys for the current process. For future launches, use the ignored file or the launching environment.

`CLICKY_ANTHROPIC_MODEL` and `CLICKY_OPENAI_MODEL` select fast text models. The corresponding `_VISION_MODEL` and `_AGENT_MODEL` variables select the stronger models for images and planning, for example `CLICKY_ANTHROPIC_VISION_MODEL`.

For offline operation on Apple Silicon, start Ollama. Install the local dependency lock and download the checkpoints once while online:

```sh
uv pip sync --python .venv-cursormix/bin/python --require-hashes cursormix/requirements-local.lock
ollama pull qwen3:4b
.venv-cursormix/bin/hf download mlx-community/FastVLM-0.5B-bf16
.venv-cursormix/bin/hf download mlx-community/whisper-small.en-mlx config.json weights.npz
```

The local path uses Qwen3-4B for text, FastVLM for screen observations and Whisper for optional dictation. **Local** sends text to Ollama on loopback. Model files stay outside the checkout.

**Settings → Dictation Language** applies to spoken questions and direct dictation. **Provider Default** keeps English for local Whisper and automatic detection for OpenAI. Choose a specific language for short utterances or **Detect Automatically** for multilingual input. For automatic detection or a language other than English with local Whisper, download the multilingual checkpoint once with `.venv-cursormix/bin/hf download mlx-community/whisper-small-mlx config.json weights.npz`.

Ask “Open the Appearance pane in System Settings” to navigate directly to a pane installed on this Mac. CursorMix checks the resulting window before reporting success.

When a desktop task types over highlighted text, the action description flags the replacement. If the editor exposes its insertion point or selection, CursorMix checks it again before typing.

For a file copy, move or rename, CursorMix prepares an action description containing the full source and destination filenames. It checks the resulting file and keeps an occupied destination intact. You can also explicitly request a supported terminal command.

### Launch from Finder

Open **dist/Clicky.app** to start the menu-bar app and its floating pointer. The app bundle contains its own Python runtime and dependencies. To build it locally from the checkout, run the following with the CursorMix environment set up as above.

```sh
.venv-cursormix/bin/python packaging/cursormix/build.py --local --use-checkout-env
open dist/Clicky.app
```

A rebuild replaces the finished app after it verifies the new copy. Quit the running copy first or use `--output /path/to/Clicky.app` for another copy. The default build includes the cloud client and local Ollama text support. `--local` also includes local vision and transcription libraries and uses the separately installed checkpoints. `--use-checkout-env` saves the location of the ignored credential file in this Mac’s preferences. The bundle contains no checkout credentials. This build is signed locally for the architecture of the build machine.

### Using CursorMix

- Ask about a visible message or selection. To ask about a passage, highlight it in the foreground app before summoning CursorMix. When the focused control supplies the selection through Accessibility, CursorMix reads its first 1,600 characters along with the current screen. It also uses selection changes to check walkthrough steps. Replies stream into the panel in one to three short sentences. Follow-up questions in the same panel include the last four exchanges. Dismissing or tapping the shortcut again starts a fresh conversation. A push-to-talk follow-up from the open panel keeps the recent exchange when you stay in the same app. To keep a conversation while switching windows, bring the new window forward and choose **Continue with Current Screen** from the CursorMix menu. This takes a fresh snapshot and releases any earlier marks.
- After an answer, the panel can show up to two suggested follow-ups. The selected provider receives a separate text-only request containing recent questions and replies. Select one to submit its displayed text. You can turn these off in **Settings → Suggest Follow-Ups**. Passing over eight sets hides suggestions for an hour.
- Ask “Annotate this Finder window. Circle the folder and draw an arrow to Search.” You can also ask CursorMix to connect two items with a labeled line. When macOS exposes every marked control, the drawing is click-through and its marks follow the controls. **Clear Guidance** removes it. For a printed label without an accessible control, CursorMix looks for its exact words using on-device text recognition. If an item is identifiable only in the image, CursorMix opens a resizable **Annotated Capture** of the screen you summoned. Close it or press Esc when done.
- Ask “Show me this folder in 3D” or “Draw a wheel and axle in 3D.” A floating drawing opens with labeled parts. Drag to rotate, scroll to zoom, and click a part or select its name to read its explanation. **Reset View** returns to the whole drawing. Close it to return to your answer.
- Choose **Mark Screen…** to circle or tap part of the summoned screen. Choose **Done** and ask about the mark. **Edit Marks…** lets you change or clear it. Drawings and the captured image stay in memory for that conversation.
- The pointer follows beside your mouse. Pick its color in **Settings → Pointer Color**. A waveform shows microphone capture, three dots show processing, and a speaker shows speech playback. Ask “Where do I click to …?” and it travels to the control and stays with its highlight. For controls exposed through macOS Accessibility, CursorMix tracks the visible part of the control as its window or scroll area moves. When visible controls share a label, describe the row or position, or choose from **Matching controls** in the panel. Click through the guide to use the control. **Clear Guidance** returns the pointer to your mouse. Use **Settings → Show Floating Pointer** to hide the companion. Ask “Walk me through …” for one instruction at a time. CursorMix chooses the next control from the current window and the actions it has verified. It remembers the observed window and selection across steps, so you can ask it to return to where you started. A saved walkthrough keeps a short excerpt of highlighted text (up to 140 characters) to help restore a passage after an interruption. When a walkthrough points to an icon that macOS doesn’t identify, CursorMix looks for the same shape and color in the app’s next image. The arrow follows a unique match and hides when the icon disappears or several copies look alike. During a walkthrough it also checks the active app’s visible window when accessibility controls stay unchanged, so a canvas or editor can show that a step is done. If another window or folder opens before the step is complete, CursorMix looks at the new view and guides you from there. It checks the requested outcome before ending the walkthrough. Type a question in **Ask about this step** and press Return to get a short answer alongside the current instruction. CursorMix keeps the last four clarification exchanges during that walkthrough. If you cannot find the control, choose **Help with Step** in the panel or **Help with This Step** from the menu. CursorMix looks at the current window again and offers another way through that step. With **Speak Replies** enabled, it also reads each instruction aloud.
- To dictate into another app, choose **Settings → Dictation Shortcut → Control–Option–D**. Place the caret or select text in an editable field, hold the shortcut, speak, and release. CursorMix inserts into the original selection using macOS Accessibility. If the field changes during transcription, the panel shows the transcript for copying. Dictation takes no screenshot and sends no Return key. The dictation shortcut is off by default and independent of the Ask shortcut.
- Ask CursorMix to operate a desktop app. It uses the summoned screenshot and Accessibility controls, can read menu commands, and asks for confirmation of a task outline before starting an ordinary request. Individual steps then run under that confirmation; use Stop in the panel to end the task. In Chrome and Safari, CursorMix can identify an open tab by name and verify that the same tab is still available before selecting it. Before reporting success, CursorMix checks the current window, observed transitions and available command results. When controls leave the outcome unclear, it can inspect the foreground app again. For a regular-file copy it can compare the source and destination bytes locally. If a control disappears before the action, CursorMix reads the updated window and can propose a different approach. **Cancel Current Request** stops further steps. The terminal action runs approved local utilities directly, including `pwd`, `ls`, `mkdir`, `mv` and `cp`. The integration registry currently ships one adapter. A compose request opens a draft through the default mail application. Gmail, Notion and Google Sheets OAuth adapters remain unimplemented.
- Enable **Push to Talk** to dictate by holding the shortcut. A tap still opens typed input. OpenAI mode can transcribe with its API key; other providers use local Whisper. **Speak Replies** enables macOS speech. In **Settings → Custom Dictation Words**, enter a name such as `CursorMix` to preserve its spelling, or `fig ma = Figma` to correct a recurring mishearing. CursorMix applies these spellings on your Mac to both dictated questions and text inserted in another app.

### Routines

Choose **CursorMix → Routines → New Routine**. Give the task a name and instructions, choose its provider, and select **On demand**, **Once**, **Daily** or **Weekly**. Set the first due date and local time. Enable **Launch at Login** if you want CursorMix to start when you sign in.

A **Background brief** runs while CursorMix is open. You can give it a UTF-8 text file up to 32 KB. Each run reads the current file and uses the saved provider. The scheduler starts up to three briefs at once. Further due briefs appear as **Queued** and start in due-time order as slots open. CursorMix Home shows the counts. Choose **Jobs** to open the routines menu while briefs are running, or **Start** when a desktop task is due. Use **Cancel Run** on an individual brief to stop it while the others continue. Open **View Result** to read the last completed brief or the reason a run stopped. **Local** uses Ollama. **Automatic** resolves the available provider at run time.

A scheduled **Desktop task** appears as **Ready** in Routines and in CursorMix Home. Bring the intended app forward, then choose **Run on Current Screen** or **Start** in Home. CursorMix captures the current screen on that summon and starts its desktop task flow. Missed occurrences coalesce into one run when CursorMix reopens or wakes. An interrupted occurrence stays marked **Interrupted** and can be started again explicitly.

The routine menu also lets you run, edit, pause, resume or remove a task. Routines and their last results are saved separately from conversation History. Removing a routine deletes its saved instructions, schedule and result.

### Permissions and privacy

Screen Recording supplies the screenshot. Accessibility supplies window text, control locations and desktop actions. Microphone access is used when recording speech. **Settings → Permissions** shows each grant and opens its System Settings pane. Reopen the launching app if a grant requires a restart.

ScreenCaptureKit captures the selected display in memory, with an MSS fallback at summon. During an active guide or desktop task, requested visual checks use ScreenCaptureKit to capture the guided application and menu bar on that display. Passive mode and nudges are off by default. Passive mode reads only the frontmost app name. When a cloud provider is selected, submitted requests can include window text and the screenshot. Disable **Use Screen Context** for text-only requests. Queued captures check the session lock before reading the window and again before returning its contents. CursorMix also discards a capture delivered to the panel while the Mac is locked.

Hover over **CursorMix** at the top center of a display to open Home. You can also pin it open from **Open CursorMix Home** in the menu. Choose **Ask** to open a conversation below that display’s menu bar. Screen context follows the frontmost app’s window. Home shows the current answer or task progress and lets you return to the request, cancel it, or open History. Opening Home leaves your active app focused. The global shortcut still opens Ask beside the pointer. To hide the tab on every display, turn off **Settings → Show CursorMix Home at Top of Display**.

Choose **History** to search saved requests, replies, app names and window titles. Select a conversation and choose **Resume** to continue from its last four saved exchanges. CursorMix takes fresh screen context when enabled. Older ungrouped entries each appear as a conversation. **Settings → Answer Style** remembers your choice of standard, plain language, answer first, or one sentence.

With **Save Request History** on, an unfinished walkthrough appears in History. Bring the app you were learning back to the front, choose the task, and click **Resume Walkthrough**. CursorMix checks the current window before it points again. It saves the goal, plan, step questions and short observations locally. Before finishing, CursorMix checks the requested result against the current window and observed changes. A round trip or repeated action needs evidence of those transitions. Finishing replaces the saved walkthrough with a conversation entry. History retention and **Clear History** apply to saved steps.

History stores short request and reply excerpts and, for screen-related turns, the observed app name and window title in `~/Library/Application Support/Clicky/history.sqlite3`. Configure retention, turn off new history entries, or clear the database from History. Temporary voice recordings are removed after transcription or cancellation.

### Verification

```sh
.venv-cursormix/bin/python -m cursormix --check-startup
.venv-cursormix/bin/python -m eval.cursormix.corpus --install latency qa router agent
uv run --no-project --python 3.12 --with pyarrow --with pillow eval/cursormix/screen2ax.py
.venv-cursormix/bin/python -m eval.cursormix.corpus --install point
.venv-cursormix/bin/python eval/cursormix/run_all.py
```

The public fixtures cover screen Q&A, latency, negative routing cases, Retina pointing, and scripted desktop task definitions. The full routing, teaching, native pointing, and paired desktop action gates require additional real observations. The suite reports missing evidence as incomplete. See the [measured results](../eval/cursormix/results.md) and [final live checks and fixture instructions](../eval/cursormix/MANUAL.md).
