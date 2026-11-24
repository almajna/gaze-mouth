# Models, assets, and recordings

Run `.venv-gf/bin/python setup_assets.py` from the repository root. Add `--planner` for the optional vision and planning models (about 3.5 GB together). Add `--check` to verify without downloading. The installer checks each artifact against [the pinned manifest](asset_manifest.json), which gives its exact destination, download URL, revision and checksum. Large files use SHA-256. Small Hugging Face files without SHA-256 metadata use the published Git blob ID. Repositories use pinned Git commit IDs. If an existing asset fails verification, the installer leaves it in place and reports the mismatch.

## Downloadable assets

| Model or asset | Install location | Upstream terms and provenance |
| --- | --- | --- |
| USR 2.0 Base+ visual speech model | `models/usr2/baseplus.pth` in the pinned `models/usr2/` checkout | [USR 2.0 code and model table](https://github.com/ahaliassos/usr2). Code MIT. Separate checkpoint terms unverified. The included `utils/labels/unigram1000_units.txt` is the tokenizer input. |
| Auto-AVSR LRS3 V WER 19.1 | `models/auto_avsr/vsr.pth` in the pinned `models/auto_avsr/` checkout | [Auto-AVSR model zoo](https://github.com/mpc001/auto_avsr#model-zoo), checkpoint `vsr_trlrs2lrs3vox2avsp_base.pth`. Code Apache-2.0. Separate checkpoint terms unverified. The checkout supplies the 68-point mean face and Auto-AVSR tokenizer. |
| MediaPipe Face Landmarker float16 v1 | `face_landmarker.task` | [Model bundle and constituent model cards](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker#models). FaceMesh model card lists Apache-2.0. Exact bundle terms unverified. |
| MediaPipe Hand Landmarker float16 v1 | `models/hand_landmarker.task` | [Hand model bundle](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker#models). MediaPipe code Apache-2.0. Exact bundle terms unverified. |
| OpenCV LBF 68-point model | `models/lbfmodel.yaml` | [Pinned GSOC2017 source](https://github.com/kurnianggoro/GSOC2017/blob/252614093b29280c549351971d44d71e63c9197c/data/lbfmodel.yaml). Artifact license unverified. |
| GazeFollower MGazeNet, BlazeFace, Face Landmarker | `.venv-gf/lib/python3.12/site-packages/gazefollower/res/model_weights/{base.mnn,blaze_face.mnn,face_landmarker.task}` | [GazeFollower](https://github.com/GanchengZhu/GazeFollower) lists CC BY-NC-SA 4.0. Bundled with the hash-pinned 1.1.0 wheel. Separate bundled model terms unverified. |
| English KenLM 4-gram and unigrams | `models/english-4gram.bin`, `models/english-4gram-unigrams.txt` | [Hugging Face upload](https://huggingface.co/patrickvonplaten/wav2vec2-base-100h-with-lm/tree/0612413f4d1532f2e50c039b2f014722ea59db4e/language_model). Exact binaries' license and derivation unverified. |
| MLX Whisper small.en and large-v3-turbo | `models/whisper-small.en-mlx/`, `models/whisper-large-v3-turbo/` | [small.en conversion](https://huggingface.co/mlx-community/whisper-small.en-mlx), [turbo conversion](https://huggingface.co/mlx-community/whisper-large-v3-turbo), [upstream Whisper MIT code and weights](https://github.com/openai/whisper#license). Separate conversion terms unverified. |
| MLX Qwen2.5-3B-Instruct 4-bit | `models/Qwen2.5-3B-Instruct-4bit/` | [Pinned conversion and Qwen Research terms](https://huggingface.co/mlx-community/Qwen2.5-3B-Instruct-4bit). Used for phoneme reconstruction and GER. |
| MLX Qwen3-4B-Instruct-2507 4-bit | `models/Qwen3-4B-Instruct-2507-4bit/` | [Apache-2.0 model card](https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit). Optional planner. |
| MLX FastVLM-0.5B bf16 | `models/FastVLM-0.5B-bf16/` | [Apple AMLR model card](https://huggingface.co/mlx-community/FastVLM-0.5B-bf16). Optional visual queries. |

Files provided by the pinned upstream checkouts and wheel have these additional SHA-256 reference values.

| Relative path | SHA-256 |
| --- | --- |
| `models/usr2/utils/labels/unigram1000_units.txt` | `ead1469024d34194e1c7abfe2ca0457f06d5afee78267af2a0cbdea946f359f4` |
| `models/auto_avsr/preparation/detectors/retinaface/20words_mean_face.npy` | `dbf68b2044171e1160716df7c53e8bbfaa0ee8c61fb41171d04cb6092bb81422` |
| `models/auto_avsr/spm/unigram/unigram5000.model` | `2c1d648ccf5fdce6612ecbea2ffbd1cab5aabc90458781186175c3911c4bdb1e` |
| `models/auto_avsr/spm/unigram/unigram5000_units.txt` | `29001d967ad290e28f6f1d7d47e07e314354bf67d3b1f494e936fa2fed9a9122` |
| GazeFollower `base.mnn` | `2f96b95275fe6d7b79e98df3237ebb96e15ef5522c96968f08167da7e1954a96` |
| GazeFollower `blaze_face.mnn` | `c25d44beefefb2e45ea3184b43a35a763c803412ffbed43e463586ec75aeed9c` |
| GazeFollower `face_landmarker.task` | `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff` |

The local `models/mobileone_s0_gaze.onnx` is unused by the current entry points (SHA-256 `8b4fdc4e3da44733c9a82e7776b411e4a39f94e8e285aee0fc85a548a55f7d9f`). [yakhyo/gaze-estimation](https://github.com/yakhyo/gaze-estimation) is a candidate origin. Exact source, release, and weight license are unverified.

## Locally generated assets

The gaze controller uses a personal calibration and trigger configuration. Mouthed command recognition uses a locally trained command model, and sentence dictation uses a speaker adapter. The training and calibration tools are included. Personal recordings, sentence lists, calibration, and trained weights are excluded from this release. Use recordings you have permission to use, and follow the terms of the upstream models and datasets when training or distributing derivatives.

The macOS audio cues use the host OS sounds from `/System/Library/Sounds/`. The installer does not bundle them.

## Dataset provenance

`data/session_*`, `data/silent_train/`, `data/silent_speech/`, `data/train_sentences/` and `data/eval_holdout/` are project recordings. Collection uses local, user-supplied prompt lists. The upstream LRS2, LRS3, VoxCeleb2 and AVSpeech corpora are model-training provenance, with no dataset copies needed for these runs. See the [Oxford LRS2](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html), [Oxford LRS3](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html), [VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) and [AVSpeech](https://looking-to-listen.github.io/avspeech/) dataset pages for their separate access and rights terms.
