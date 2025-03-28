from dataclasses import dataclass
from typing import List, Tuple

import torch
import torchaudio
from huggingface_hub import hf_hub_download
from models import Model
from moshi.models import loaders
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer
from watermarking import CSM_1B_GH_WATERMARK, load_watermarker, watermark
import time

@dataclass
class Segment:
    speaker: int
    text: str
    # (num_samples,), sample_rate = 24_000
    audio: torch.Tensor


def load_llama3_tokenizer():
    """
    https://github.com/huggingface/transformers/issues/22794#issuecomment-2092623992
    """
    tokenizer_name = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[
            (f"{bos}", tokenizer.bos_token_id),
            (f"{eos}", tokenizer.eos_token_id),
        ],
    )
    return tokenizer


class Generator:
    def __init__(self, model: Model):
        self._model = model
        self._model.setup_caches(1)
        self._text_tokenizer = load_llama3_tokenizer()
        self.device = next(model.parameters()).device
        self._audio_tokenizer = self._load_audio_tokenizer()
        self._watermarker = load_watermarker(device=self.device)
        self.sample_rate = self._audio_tokenizer.sample_rate
        self.ctx_tokens = []
        self.ctx_tokens_mask = []

    def _load_audio_tokenizer(self):
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        mimi = loaders.get_mimi(mimi_weight, device=self.device)
        mimi.set_num_codebooks(32)
        return mimi

    @torch.inference_mode()
    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
        text_frame = torch.zeros(len(text_tokens), 33).long()
        text_frame_mask = torch.zeros(len(text_tokens), 33).bool()
        text_frame[:, -1] = torch.tensor(text_tokens)
        text_frame_mask[:, -1] = True
        return text_frame.to(self.device), text_frame_mask.to(self.device)

    @torch.inference_mode()
    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert audio.ndim == 1, "Audio must be single channel"
        # (K, T)
        audio = audio.to(self.device)
        audio_tokens = self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]

        eos_frame = torch.zeros(audio_tokens.size(0), 1).to(self.device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

        audio_frame = torch.zeros(audio_tokens.size(1), 33).long().to(self.device)
        audio_frame_mask = torch.zeros(audio_tokens.size(1), 33).bool().to(self.device)
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :-1] = True

        return audio_frame, audio_frame_mask

    @torch.inference_mode()
    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)
        return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_masks, audio_masks], dim=0)

    @torch.inference_mode()
    def update_ctx_tokens(self, context: List[Segment]):
        start_time = time.time()
        self.ctx_tokens, self.ctx_tokens_mask = zip(*[self._tokenize_segment(seg) for seg in context]) if context else ([], [])
        duration = (time.time() - start_time)
        print(f"update_ctx_tokens: {duration*1000:.02f} ms")

    @torch.inference_mode()
    def _prepare_prompt_tokens(self, text: str, speaker: int, context: List[Segment]):
        tokens, tokens_mask = (self.ctx_tokens, self.ctx_tokens_mask)
        start_time = time.time()
        gen_tokens, gen_masks = self._tokenize_text_segment(text, speaker)
        duration = (time.time() - start_time)
        print(f"_prepare_prompt_tokens: text: {duration*1000:.02f} ms")
        return (
            torch.cat([*tokens, gen_tokens], dim=0).long().to(self.device),
            torch.cat([*tokens_mask, gen_masks], dim=0).bool().to(self.device),
        )

    @torch.inference_mode()
    def generate_stream(
            self,
            text: str,
            speaker: int,
            context: List[Segment],
            max_audio_length_ms=90_000,
            temperature=0.9,
            topk=50):
        self._model.reset_caches()
        max_generation_len = int(max_audio_length_ms / 80)
        prompt_tokens, prompt_tokens_mask = self._prepare_prompt_tokens(text, speaker, context)

        curr_tokens, curr_tokens_mask = prompt_tokens.unsqueeze(0), prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)

        max_seq_len = 2048
        max_context_len = max_seq_len - max_generation_len
        if curr_tokens.size(1) >= max_context_len:
            raise ValueError(
                f"Inputs too long, must be below max_seq_len - max_generation_len: {max_context_len}"
            )

        with self._audio_tokenizer.streaming(1):

            # Frame duration in seconds
            frame_duration = 0.080  # 80 ms
            sample_rate = self.sample_rate  # 24,000 Hz
            for i in range(max_generation_len):
                start_time = time.time()
                sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)

                if torch.all(sample == 0):
                    break  # EOS

                yield self._audio_tokenizer.decode(sample.unsqueeze(-1)).squeeze().unsqueeze(1)

                if i > 50:
                    # Calculate elapsed time and sleep to match real-time playback
                    elapsed_time = time.time() - start_time
                    sleep_time = max(0, frame_duration - elapsed_time)
                    time.sleep(sleep_time)

                curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
                curr_tokens_mask = torch.cat([torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1).unsqueeze(1)
                curr_pos = curr_pos[:, -1:] + 1

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms=90_000,
        temperature=0.9,
        topk=50
        ) -> torch.Tensor:

        samples = self.generate_stream(
            text,
            speaker,
            context,
            max_audio_length_ms,
            temperature,
            topk)
        
        audio = torch.cat(list(samples))

        # This applies an imperceptible watermark to identify audio as AI-generated.
        # Watermarking ensures transparency, dissuades misuse, and enables traceability.
        # Please be a responsible AI citizen and keep the watermarking in place.
        # If using CSM 1B in another application, use your own private key and keep it secret.
        audio, wm_sample_rate = watermark(self._watermarker, audio, self.sample_rate, CSM_1B_GH_WATERMARK)
        return torchaudio.functional.resample(audio, orig_freq=wm_sample_rate, new_freq=self.sample_rate)


def load_csm_1b(device: str = "cuda") -> Generator:
    model = Model.from_pretrained("sesame/csm-1b").to(device=device, dtype=torch.bfloat16)
    return Generator(model)
