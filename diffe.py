from huggingface_hub import hf_hub_download
from generator import load_csm_1b, Segment
import torchaudio
import traceback
from torch import backends, cuda, cat as torch_cat
import os
from run_csm import prepare_prompt

os.environ["NO_TORCH_COMPILE"] = "1" # Probably optional, but disables mimi lazy compile and disables the need for triton.

try:
    print("Loading model...")
    model_path = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt")
    if backends.mps.is_available():
        device = "mps"
    elif cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    generator = load_csm_1b(device)
    print("Model loaded successfully!")
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
    if input("Would you like to test if your audio was successfully loaded with a quick gen? (y/N): ").lower() == "y":
                print("Generating audio for: 'This is a test of the reference audio for speaker 0.'")
                audio = generator.generate(
                    text="This is a test of the reference audio for speaker 0.",
                    speaker=0,
                    context=context_segments,
                    max_audio_length_ms=25_000,
                    # temperature=0.95,
                    # topk=3,
                    # max_seq_len=4096
                )
                print("Saving audio to audioTest.wav...")
                torchaudio.save("audioTest.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
                print("Audio saved successfully!")
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
                
                audio = generator.generate(
                    text=text if not pwii else pwii[0],
                    speaker=spkr,
                    context=context_segments,
                    max_audio_length_ms=25_000,
                    # temperature=0.95,
                    # topk=3,
                    # max_seq_len=4096 # uncomment any of these to use/change them as needed. For this one, please remember to change it for both models in models.py as well.
                )
                
                context_segments.append(
                    Segment(text=text if not pwii else pwii[0], speaker=spkr, audio=audio)
                )
                print("Saving audio to audio.wav...")
                torchaudio.save("audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
                print("Audio saved successfully!")
                    
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
                    
                        audio = generator.generate(
                            text=pwii[i],
                            speaker=spkr + spkrOffset,
                            context=prompt_segments + generated_segments,
                            max_audio_length_ms=25_000,
                            # max_seq_len=4096
                        )
                        
                        generated_segments.append(
                            Segment(text=pwii[i], speaker=spkr + spkrOffset, audio=audio)
                        )

                        print(f"Saving audio to audio{i}.wav...")
                        torchaudio.save(f"audio{i}.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
                            
                        conversation_history.append({"role": "user", "content": pwii[i]})
                    print("Fusing audio...")
                    all_audio = torch_cat([seg.audio for seg in generated_segments], dim=0)
                    print("Saving fused audio to outputCombined.wav...")
                    torchaudio.save("outputCombined.wav", all_audio.unsqueeze(0).cpu(), generator.sample_rate)
            
        except Exception as e:
            print(f"Error generating audio: {e}")
            traceback.print_exc()
except Exception as e:
    print(f"Error initializing model: {e}")
    traceback.print_exc() 
