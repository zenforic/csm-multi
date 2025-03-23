# CSM-Multi

## Diffe
**diffe.py** Modified script from issue [61](https://github.com/SesameAILabs/csm/issues/61) from the original Sesame repo. Thanks to [@asiff00](https://github.com/asiff00) for the original code, there is now a way to, while keeping the models loaded, enter multiple text prompts without waiting for a reload each time. I've also added multi speaker support using speaker offset text splitting. simply separate each speaker's text with `||` and add a number to the end of the text that will be used as the offset (else a default offset will be applied). It also included a simple reference audio input capability (needs to be mono, not stereo), does not have silence processing as in [csm-voice-cloning](https://github.com/isaiahbjork/csm-voice-cloning) repo's voice-clone.py, of which I've also added a modification for. (Thanks to that repo's author for the original, most use instructions can be found there, but new things will be discussed here).

~~The maximum amount of "multi-speakers" for now is 4, but this can be experimented with in the code. I've not yet come up with a better solution to ffmpeg concat argument increases per gen.~~

Using the torch catonate method referenced in the new `run_csm.py` example from the original repo, the above limitation has been removed as well as Diffe's reference/audio prompting capability greatly improved. For now, the old method is still used in `voice_clone.py` until all the new techniques are possibly later applied there.

### References usage:

By default, Diffe will load `reference.wav` as an audio prompt/reference. It is loaded into speaker id 0, which Diffe defaults to using (see [Commands Usage](https://github.com/zenforic/csm-multi?tab=readme-ov-file#commands-usage)). It will now also load (and require) an audio transcription (this can be done with Whisper) of said reference audio in the same name with .txt instead (i.e. `reference.txt`). Subsequently number-suffixed filenames with the same prefixed name (`reference` in this case, so `reference1.wav`, `reference2.wav`, etc.) will be loaded and require similarly named txt transcripts to use. Each will be sorted as done by Python's list `sort()` function and subsequently placed in the speaker numbers/IDs in order. So to switch to `reference1.wav`'s speaker, use `$SWAP$` to use it directly or add a 1 to the end of the sentence in whichever offset separated multi-conversation (see below) you're doing. Any speaker numbers used past the known references will highly likely be an entirely new speaker.

### Offsets usage:
Example input:
`Yeah, they're real cool.||I see that Yala! How rare are they?||I dunno, they're a pwii.0`

The `||`s are the separators for different generations in one go (4 max) and the option `0` is the offset to the speaker, which defaults to +1 per `||`. In this case, the first speaker will speak, then the second, then the first again.

## Voice-Clone

**3/23/2025**: This is now slightly superceded by Diffe's reference capability which was increased thanks to recent commits to the original repo. This will still be kept around as it can and likely will be fixed in future since it has [long] silence removal processing.

Thanks to the efforts in this repo: [csm-voice-cloning](https://github.com/isaiahbjork/csm-voice-cloning) a "better than simple reference" voice cloning via Sesame exists. See the original repo for more. New features added are the same multispeaker functionality and commands from diffe.

To use the 4096 (or other context sizes) rename `models.py` to `models.py.old` or some such and rename `models-cloning.py` to `models.py` and follow the instructions in the original csm-voice-cloning repo on how to change that and the `voice_clone.py` file for more context.

## Commands usage
Both scripts contain commands you can enter in the enter text prompt:

* `$CLEAR$` clear the context. Useful if generations are getting weird or errors about the inputs being too long crop up.
* `$SWAP$` Swap primary speakers (increments `spkr` by 1).
* `$BACK$` Swap backwards on primary speakers (decrements `spkr` by 1).
* `$HISTORY$` View conversation history (text).
* `$EXIT$` Gracefully exit the script.

# CSM

**2025/03/13** - We are releasing the 1B CSM variant. The checkpoint is [hosted on Hugging Face](https://huggingface.co/sesame/csm_1b).

---

CSM (Conversational Speech Model) is a speech generation model from [Sesame](https://www.sesame.com) that generates RVQ audio codes from text and audio inputs. The model architecture employs a [Llama](https://www.llama.com/) backbone and a smaller audio decoder that produces [Mimi](https://huggingface.co/kyutai/mimi) audio codes.

A fine-tuned variant of CSM powers the [interactive voice demo](https://www.sesame.com/voicedemo) shown in our [blog post](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice).

A hosted [Hugging Face space](https://huggingface.co/spaces/sesame/csm-1b) is also available for testing audio generation.

## Requirements

* A CUDA-compatible GPU
* The code has been tested on CUDA 12.4 and 12.6, but it may also work on other versions
* Similarly, Python 3.10 is recommended, but newer versions may be fine
* For some audio operations, `ffmpeg` may be required
* Access to the following Hugging Face models:
  * [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)
  * [CSM-1B](https://huggingface.co/sesame/csm-1b)

### Setup

```bash
git clone git@github.com:SesameAILabs/csm.git
cd csm
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Disable lazy compilation in Mimi
export NO_TORCH_COMPILE=1

# You will need access to CSM-1B and Llama-3.2-1B
huggingface-cli login
```

### Windows Setup

The `triton` package cannot be installed in Windows. Instead use `pip install triton-windows`.

## Usage

Run the example script:
```bash
python run_csm.py
```
You can also create your own script using the example code below.

Generate a sentence

```python
from generator import load_csm_1b
import torchaudio
import torch

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

generator = load_csm_1b(device=device)

audio = generator.generate(
    text="Hello from Sesame.",
    speaker=0,
    context=[],
    max_audio_length_ms=10_000,
)

torchaudio.save("audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
```

CSM sounds best when provided with context. You can prompt or provide context to the model using a `Segment` for each speaker's utterance.

```python
from generator import Segment

speakers = [0, 1, 0, 0]
transcripts = [
    "Hey how are you doing.",
    "Pretty good, pretty good.",
    "I'm great.",
    "So happy to be speaking to you.",
]
audio_paths = [
    "utterance_0.wav",
    "utterance_1.wav",
    "utterance_2.wav",
    "utterance_3.wav",
]

def load_audio(audio_path):
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=generator.sample_rate
    )
    return audio_tensor

segments = [
    Segment(text=transcript, speaker=speaker, audio=load_audio(audio_path))
    for transcript, speaker, audio_path in zip(transcripts, speakers, audio_paths)
]
audio = generator.generate(
    text="Me too, this is some cool stuff huh?",
    speaker=1,
    context=segments,
    max_audio_length_ms=10_000,
)

torchaudio.save("audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
```

## FAQ

**Does this model come with any voices?**

The model open-sourced here is a base generation model. It is capable of producing a variety of voices, but it has not been fine-tuned on any specific voice.

**Can I converse with the model?**

CSM is trained to be an audio generation model and not a general-purpose multimodal LLM. It cannot generate text. We suggest using a separate LLM for text generation.

**Does it support other languages?**

The model has some capacity for non-English languages due to data contamination in the training data, but it likely won't do well.

## Misuse and abuse ⚠️

This project provides a high-quality speech generation model for research and educational purposes. While we encourage responsible and ethical use, we **explicitly prohibit** the following:

- **Impersonation or Fraud**: Do not use this model to generate speech that mimics real individuals without their explicit consent.
- **Misinformation or Deception**: Do not use this model to create deceptive or misleading content, such as fake news or fraudulent calls.
- **Illegal or Harmful Activities**: Do not use this model for any illegal, harmful, or malicious purposes.

By using this model, you agree to comply with all applicable laws and ethical guidelines. We are **not responsible** for any misuse, and we strongly condemn unethical applications of this technology.

---

## Authors
Johan Schalkwyk, Ankit Kumar, Dan Lyth, Sefik Emre Eskimez, Zack Hodari, Cinjon Resnick, Ramon Sanabria, Raven Jiang, and the Sesame team.