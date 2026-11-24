# Gaze + Mouth

**Experimental hands-free interaction for macOS:** use eye gaze, mouth gestures, silent speech and voice to explore controlling a computer. The repository also includes a separate screen-aware desktop assistant called CursorMix.

This is a research prototype, not an accessibility product ready for daily use. It was developed and evaluated primarily on an Apple Silicon Mac with a small set of personal recordings. Accuracy varies by person and session; some capabilities still require live validation.

## What's here

| Prototype | What it explores | Entry point |
| --- | --- | --- |
| **Hands-free Mac control** | Calibrated webcam gaze, jaw-triggered clicking, hand gestures, gaze-paced scrolling and spoken or mouthed commands. | `handsfree_mac_agent.py` |
| **Silent speech** | Webcam mouth crops and a speaker-adapted visual speech model for sentence transcription. | `live_lipread.py` |
| **CursorMix assistant** | A separate menu-bar app that answers questions about the screen, points at controls, guides workflows and proposes desktop actions. Supports voice, history and scheduled routines. | `python -m cursormix` |

CursorMix is an independent implementation inspired by [the original open-source project](https://github.com/farzaa/clicky); the upstream project is separate and is not included in this repository.

## Results and limitations

These are recorded experiment results, not guarantees. The release includes source, public evaluation cases, and aggregate results. Personal recordings, transcripts, calibration files, and locally trained weights are kept private. See the linked reports for methods and sample sizes.

| Experiment | Failure modes and generalization first | Other observations |
| --- | --- | --- |
| **Gaze and jaw click** | The saved jaw trigger fired on **0/27 negative clips**. Cross-session gaze quadrant error was **19.06–30.70% per frame** across six calibration/test directions. | See the [detailed guide](docs/USER_GUIDE.md#recorded-results) for clip-level results and false-scroll behavior. |
| **Silent speech** | The speech gate returned text on **0/64 non-speech clips**, but those clips were used to select the threshold. Cross-session sentence word error rate (WER) is unmeasured. | On 50 held-out sentences from one speaker and one session, speaker adaptation reduced WER from **89.78% to 62.37%**. The full stack reached **59.41%**, still far from reliable dictation. [Experiment log](docs/plan.md) |
| **CursorMix assistant** | Claude routed **0/76 classified negative requests** to desktop actions. Human cross-session routing error is unmeasured. On the evaluated build, paired live desktop task trials were unavailable. | An earlier evaluated revision answered **44/50** screen questions and placed **92/100** pointers in the target. It verified **6/24** teaching steps across three recorded workflows. [Evaluation report](eval/cursormix/results.md) |

The assistant measurements are from an earlier private revision. They describe the original evaluation, which is broader than the fixtures in this release. False-positive results come first because unintended actions matter in a hands-free interface.

## Try the screen-aware assistant

On macOS, install Python 3.12 and [uv](https://docs.astral.sh/uv/), then from the project directory:

```sh
uv venv --python 3.12 .venv-cursormix
uv pip sync --python .venv-cursormix/bin/python --require-hashes cursormix/requirements.lock
.venv-cursormix/bin/python -m cursormix
```

Press **Control–Option–Space**, then ask a question about the current screen. The app requests macOS permissions for the features used. Configure an Anthropic or OpenAI key in an ignored `.env.local` file or through the launching environment. There is also an Apple Silicon local-model option; it requires Ollama and separately downloaded models. See [assistant setup and usage](docs/USER_GUIDE.md#cursormix-for-macos).

Screen context sent to a cloud provider can include a screenshot and window text. Conversation excerpts can be stored locally; settings control history and screen context. More details are in the [privacy notes](docs/USER_GUIDE.md#permissions-and-privacy).

## Explore hands-free control and lip reading

The gaze and speech prototypes require an Apple Silicon Mac, a webcam, macOS permissions, Python 3.12, dependencies and model assets. Some paths also require personal calibration or trained adapters that are not distributed here. Start with the [setup and live guide](docs/USER_GUIDE.md#set-up-on-apple-silicon) and [asset inventory](ASSETS.md); the commands below start live camera sessions after setup:

```sh
.venv-gf/bin/python handsfree_mac_agent.py --onboarding --silent-only
.venv-gf/bin/python live_lipread.py --device mps
```

The project uses real recorded data for its measurements. A fresh checkout does not include those recordings, local calibration or checkpoints, so the full evaluation suite cannot be reproduced without supplying them. The guide includes [offline replay commands](docs/USER_GUIDE.md#replay-the-recorded-checks).

## How it fits together

The hands-free controller combines gaze estimation, facial landmarks and a jaw trigger with macOS Accessibility targets. Separate visual-speech and audio pipelines handle mouthed and spoken input. An optional local planner uses visual and accessibility context to propose actions. See [architecture](ARCHITECTURE.md) for the component diagram.

| Directory | What's inside |
| --- | --- |
| [`handsfree/`](handsfree/) | Gaze, audio input, and macOS Accessibility code shared by the hands-free tools |
| [`silent_avsr/`](silent_avsr/) and [`context_decoder/`](context_decoder/) | Visual speech models, decoding, and text correction |
| [`cursormix/`](cursormix/) | Menu-bar assistant, capture, guidance, and provider clients |
| [`training/`](training/) | Data collection and model training commands |
| [`eval/`](eval/) and [`tests/`](tests/) | Recorded-data evaluation and automated checks |
| [`experiments/`](experiments/) | Earlier gaze demos. The circle and hybrid demos were written for EyeTrax 0.4.0, which is outside the main dependency lock. |
| [`packaging/cursormix/`](packaging/cursormix/) | macOS app bundle builder |

## Documentation

- [Setup, controls and recorded measurements](docs/USER_GUIDE.md)
- [Model, dataset and asset provenance](ASSETS.md)
- [Hands-free architecture](ARCHITECTURE.md)
- [Experiment methods and results](docs/plan.md)
- [Assistant evaluation results](eval/cursormix/results.md) and [live checks](eval/cursormix/MANUAL.md)
- [Résumé description](docs/RESUME.md)

## License and attribution

Project source and documentation are provided under the [MIT license](LICENSE). Third-party dependencies, checkpoints, datasets and recordings have their own terms; the MIT license does not override them. See [ASSETS.md](ASSETS.md) before using or redistributing models or derived weights. The assistant's public evaluation materials include separately attributed third-party content; see the [corpus notes](eval/cursormix/catalog/README.md).
