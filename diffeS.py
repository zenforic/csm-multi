from huggingface_hub import hf_hub_download
from generator_stream import load_csm_1b, Segment
import torchaudio
import traceback
import torch
import os
from run_csm import prepare_prompt
import sounddevice as sd
import numpy as np
import queue
from time import time

os.environ["NO_TORCH_COMPILE"] = "1" # Probably optional, but disables mimi lazy compile and disables the need for triton.

FRAME_SIZE = 1920
BUFFER_SIZE = 4

def print_time(start_time, msg):
    duration = (time() - start_time)
    if duration >=1:
        print(f"{msg} {duration:.02f} s")
    else:
        print(f"{msg} {duration*1000:.02f} ms")

class AudioStream:
    def __init__(self, samplerate):
        self.q = queue.Queue(maxsize=40)
        self.is_prebuffering = True
        self.prebuffer_size = BUFFER_SIZE
        self.buffer = []
        self.frames_in_queue = 0
        self.stream = sd.OutputStream(
            samplerate=samplerate,
            blocksize=FRAME_SIZE,
            channels=1,
            callback=self.callback
        )

    @torch.inference_mode()
    def callback(self, outdata, frames, time, status):
        if self.q.qsize() != self.frames_in_queue:
            self.frames_in_queue = self.q.qsize()
            # print(f"{self.frames_in_queue} frames in queue.")
            if self.start_time:
                print_time(self.start_time, "Time to first frame ready for playback:")
                self.start_time = None

        if self.is_prebuffering:
            if self.q.qsize() < self.prebuffer_size:
                outdata[:] = 0
                return
            else:
                self.is_prebuffering = False
        try:
            new_frame = self.q.get_nowait().unsqueeze(1)
            if new_frame.shape[0] < outdata.shape[0]:
                padding = torch.zeros(outdata.shape[0] - new_frame.shape[0], 1)
                new_frame = torch.vstack((new_frame, padding))

            outdata[:] = new_frame.numpy()
        except queue.Empty:
            outdata[:] = 0
            self.is_prebuffering = True

    def start(self):
        self.stream.start()
        return self.q

try:
    print("Loading model...")
    start_time = time()
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    generator = load_csm_1b(device)
    print("Model loaded successfully!")
    print_time(start_time, "load_csm_1b() time:")
    reference_path = "reference.wav"
    context_segments = []
    spkr = 0
    conversation_history = []
    
    def initializeContext(context_segments: list, reference_path: str, conversation_history: list, generator):
        if os.path.exists(reference_path):
            """
            def load_audio(audio_path):
                audio_tensor, sample_rate = torchaudio.load(audio_path)
                audio_tensor = torchaudio.functional.resample(
                    audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=generator.sample_rate
                )
                return audio_tensor
            """ # OLD function. Replaced with load_prompt_audio in run_csm.py during call to prepare_prompt.

            referencePrefix = reference_path.removesuffix(".wav")

            def read_reference_prompt(ref_path: str) -> str:
                if os.path.exists(prompt_path := ref_path.replace(".wav", ".txt")):
                    with open(prompt_path, "r") as f:
                        return f.read().strip()
                else:
                    return ""

            # start loading any reference audio including extra files suffixed with speaker number.
            print("Loading reference audio...")
            reference_prompts = []
            reference_audio_paths = [p for p in os.listdir() if p.endswith(".wav") and referencePrefix in p and p != reference_path]
            reference_audio_paths.sort()
            reference_audio_paths.insert(0, reference_path)
            if len(reference_audio_paths) > 5:
                print("\nThere may be too many reference audio files, this may cause out of memory errors.\n"
                "If you have the VRAM, you can uncomment and increase the max_seq_len parameter in both generator.generate()s to fix this.\n"
                "Don't forget to restart the script after changing the parameter and changing them in models.py.\n")
            print(f"Using {reference_path} as reference audio...")
            reference_prompts.append(read_reference_prompt(reference_path))
            context_segments.append(prepare_prompt(reference_prompts[0], 0, reference_path, generator.sample_rate))
            for i in range(1, len(reference_audio_paths)):
                if (reference_text := read_reference_prompt(reference_audio_paths[i])) == "":
                    print(f"Skipping reference audio {reference_audio_paths[i]} due to missing prompt file")
                    continue
                reference_prompts.append(reference_text)
                print(f"Using {reference_audio_paths[i]} as reference audio...")
                context_segments.append(prepare_prompt(reference_prompts[i], i, reference_audio_paths[i], generator.sample_rate))
            
            print("All reference audio loaded and added to context")
        else:
            print("No reference audio.wav found, starting without context")

        conversation_history = []
        if context_segments:
            for segment in context_segments:
                conversation_history.append({"role": "assistant", "content": segment.text})

    initializeContext(context_segments, reference_path, conversation_history, generator)
    num_refs = len(context_segments)
    print(num_refs)
    print("Starting audio stream...")
    audio_stream = AudioStream(generator.sample_rate)
    audio_queue = audio_stream.start()
    print("Audio stream started, ready for input.")
    if input("Would you like to test if your audio was successfully loaded with a quick gen? (y/N): ").lower() == "y":
                print("Generating audio for: 'This is a test of the reference audio for speaker 0.'")
                audio_stream.start_time = time()
                audio = generator.generate_stream(
                    text="This is a test of the reference audio for speaker 0.",
                    speaker=0,
                    context=context_segments,
                    max_audio_length_ms=25_000,
                    # temperature=0.95,
                    # topk=27,
                    # max_seq_len=4096
                )
                print_time(audio_stream.start_time, "generate_stream() time:")
                print("Playing audio...")
                for frame in audio:
                    cpu_frame = frame.detach().cpu()
                    audio_queue.put(cpu_frame)
                print("Audio played successfully!")
    running = True
    while running:
        try:
            text = input("Enter text: ")
            if (text == "$CLEAR$"):
                print("Clearing context...")
                conversation_history = []
                context_segments = []
                if os.path.exists(reference_path):
                    initializeContext(context_segments, reference_path, conversation_history, generator)
                print("Cleared.")
            elif (text == "$SWAP$"):
                spkr += 1
                print (f"Speaker set to {spkr}")
            elif (text == "$BACK$"):
                spkr -= 1
                print (f"Speaker set to {spkr}")
            elif (text == "$HISTORY$"):
                print("Conversation history:")
                for i, entry in enumerate(conversation_history):
                    print(f"{entry['role']}: {entry['content']}")
            elif (text == "$EXIT$"):
                print("Exiting...")
                running = False
            else:
                pwii = []
                if ("||" in text):
                    pwii = text.split('||')
                print(f"Generating audio for: '{text if not pwii else pwii[0]}'")
                
                audio_stream.start_time = time()
                audio = generator.generate_stream(
                    text=text if not pwii else pwii[0],
                    speaker=spkr,
                    context=context_segments,
                    max_audio_length_ms=25_000,
                    # temperature=0.95,
                    # topk=27,
                    # max_seq_len=4096 # uncomment any of these to use/change them as needed. For this one, please remember to change it for both models in models.py as well.
                )
                print_time(audio_stream.start_time, "generate_stream() time:")
                print("Playing audio...")
                for frame in audio:
                    cpu_frame = frame.detach().cpu()
                    audio_queue.put(cpu_frame)
                print("Audio played successfully!")
                    
                conversation_history.append({"role": "user", "content": text if not pwii else pwii[0]})
                
                if (pwii):
                    generated_segments = [context_segments[-1]]
                    prompt_segments = list(context_segments[:num_refs])
                    for i in range(1, len(pwii)):
                        print(f"Generating audio for: '{pwii[i]}'")
                        
                        try:
                            spkrOffset = int((pwii[i][-2:]) if pwii[i][-2] == '-' else pwii[i][-1])
                            print(f"Speaker offset: {spkrOffset}")
                            pwii[i] = pwii[i][:-2] if pwii[i][-2] == '-' else pwii[i][:-1]
                        except:
                            spkrOffset = i
                            print(f"No speaker offset found, using default: {spkrOffset}")
                        audio_stream.start_time = time()
                        audio = generator.generate_stream(
                            text=pwii[i],
                            speaker=spkr + spkrOffset,
                            context=prompt_segments + generated_segments,
                            max_audio_length_ms=25_000,
                            # max_seq_len=4096
                        )
                        print_time(audio_stream.start_time, "generate_stream() time:")

                        print("Playing audio...")
                        for frame in audio:
                            cpu_frame = frame.detach().cpu()
                            audio_queue.put(cpu_frame)
                        print("Audio played successfully!")
                        print(f"Saving audio to audio{i}.wav...")
                            
                        conversation_history.append({"role": "user", "content": pwii[i]})
            
        except Exception as e:
            print(f"Error generating audio: {e}")
            traceback.print_exc()
except Exception as e:
    print(f"Error initializing model: {e}")
    traceback.print_exc() 
