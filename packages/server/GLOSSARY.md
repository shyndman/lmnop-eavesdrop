# Server

English speech recognition and completion of transcript segments.
Shared result terms are defined in [Wire](../wire/GLOSSARY.md).

## Language

### Speech recognition

**Speech recognition**:
The conversion of recorded speech into text.

**Recognition model**:
The trained model used to recognize speech.

**Recognition backend**:
The engine that runs the recognition model.

**Inference**:
One execution of the recognition model on audio to produce recognized text.

**Voice activity detection (VAD)**:
The identification of intervals containing speech in recorded audio.

**Speech segment**:
An interval of the original recording identified as containing speech.

**Silence**:
An interval in which VAD detects no speech.

**Silence-based completion**:
Automatic completion of a segment after VAD detects enough silence after speech.

**Silence completion threshold**:
The duration of detected silence after speech required for silence-based completion.

### Audio timing and buffering

**Recording timeline**:
Elapsed time from the beginning of the original recording, including
silence between speech segments.
_Avoid_: Speech-only timeline.

**Audio buffer**:
Accepted audio retained for recognition and revision of incomplete transcription.

**Audio chunk**:
A portion of buffered audio submitted for one recognition pass.

**Buffer cursor**:
The position on the recording timeline through which buffered audio has been consumed.

**Queued audio**:
Accepted audio waiting for earlier transcription work to finish.

### Retained data

**Model cache**:
Recognition model assets or loaded models retained for reuse.
