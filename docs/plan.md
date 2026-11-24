# Experiment methods and results

This page summarizes the recorded experiments behind [Gaze + Mouth](../README.md). A public checkout includes the inference, training, and evaluation code. It requires separate, authorized recordings and assets to reproduce the personal results.

## Gaze and gesture control

The recorder stores paired webcam frames and MediaPipe landmark observations. A session includes corner fixations, deliberate jaw gestures, and negatives such as blinks, swallows, and looking down. The offline evaluation holds out repetitions and sessions, checks for unintended triggers first, and only exports a trigger configuration when the verification rules pass.

The saved jaw trigger fired on 0 of 27 negative clips. Across six calibration and test directions, gaze quadrant error ranged from 19.06% to 30.70% per frame and from 16.67% to 30.77% per clip. The [user guide](USER_GUIDE.md#recorded-results) reports false scrolls and the asymmetry between up and down scrolling.

## Visual speech

A sentence recorder saves mouth crops on a 25 Hz grid. The speech pipeline uses USR 2.0, compares joint and greedy decoding, and evaluates speaker adaptation and optional text correction. The short command classifier is evaluated separately from sentence dictation.

The fitted non-speech gate returned text on 0 of 64 clips, down from 63 of 64 without the gate. Those clips informed the threshold. Sentence WER across recording sessions is unmeasured. On the 50-clip holdout, zero-shot WER was 89.78%, the speaker adapter reached 62.37%, and adding candidate selection reached 59.41%. The small holdout and single speaker limit what these numbers show. See the [replay commands](USER_GUIDE.md#replay-the-recorded-checks).

## CursorMix

The evaluation separates asking, pointing, teaching, and running desktop tasks. It checks routing into desktop actions on negative requests and verification against recorded screen changes. See the [CursorMix evaluation report](../eval/cursormix/results.md) for the aggregate measurements and remaining gates.
