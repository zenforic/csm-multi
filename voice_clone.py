import os
from pathlib import Path
import torch
import torchaudio
from huggingface_hub import hf_hub_download
import numpy as np
import sys
import io
import traceback
from subprocess import run
import shlex


def generate_speech_with_context(
    text="Hello, this is my voice speaking with context.",
    speaker_id=999,
    context_audio_path="audio.mp3",
    context_text="This is a sample of my voice for context.",
    output_filename="output_with_context.wav"
):
    # Import the generator
    from generator import load_csm_1b, Segment
    
    # Download the model if not already cached
    model_path = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt")
    spkr = 0
    
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    # Load the model
    generator = load_csm_1b(model_path, device)
    print("Model loaded successfully!")
    print(f"Generating speech with context")
    print(f"Context audio: {context_audio_path}")
    print(f"Text: {text}")
    
    # Load context audio
    context_audio, sr = torchaudio.load(context_audio_path)
    context_audio = context_audio.mean(dim=0)  # Convert to mono
    
    # Resample if needed
    if sr != generator.sample_rate:
        context_audio = torchaudio.functional.resample(
            context_audio, orig_freq=sr, new_freq=generator.sample_rate
        )
    
    # Normalize audio volume for better consistency
    context_audio = context_audio / (torch.max(torch.abs(context_audio)) + 1e-8)
    
    # Improved silence removal that handles internal silences
    def remove_silence(audio, threshold=0.01, min_silence_duration=0.2, sample_rate=24000):
        # Convert to numpy for easier processing
        audio_np = audio.cpu().numpy()
        
        # Calculate energy
        energy = np.abs(audio_np)
        
        # Find regions above threshold (speech)
        is_speech = energy > threshold
        
        # Convert min_silence_duration to samples
        min_silence_samples = int(min_silence_duration * sample_rate)
        
        # Find speech segments
        speech_segments = []
        in_speech = False
        speech_start = 0
        
        for i in range(len(is_speech)):
            if is_speech[i] and not in_speech:
                # Start of speech segment
                in_speech = True
                speech_start = i
            elif not is_speech[i] and in_speech:
                # Potential end of speech segment
                # Only end if silence is long enough
                silence_count = 0
                for j in range(i, min(len(is_speech), i + min_silence_samples)):
                    if not is_speech[j]:
                        silence_count += 1
                    else:
                        break
                
                if silence_count >= min_silence_samples:
                    # End of speech segment
                    in_speech = False
                    speech_segments.append((speech_start, i))
        
        # Handle case where audio ends during speech
        if in_speech:
            speech_segments.append((speech_start, len(is_speech)))
        
        # Concatenate speech segments
        if not speech_segments:
            return audio  # Return original if no speech found
        
        # Add small buffer around segments
        buffer_samples = int(0.05 * sample_rate)  # 50ms buffer
        processed_segments = []
        
        for start, end in speech_segments:
            buffered_start = max(0, start - buffer_samples)
            buffered_end = min(len(audio_np), end + buffer_samples)
            processed_segments.append(audio_np[buffered_start:buffered_end])
        
        # Concatenate all segments
        processed_audio = np.concatenate(processed_segments)
        
        return torch.tensor(processed_audio, device=audio.device)
    
    # Apply improved silence removal with slightly more aggressive settings for longer files
    audio_duration_sec = len(context_audio) / generator.sample_rate
    print(f"Original audio duration: {audio_duration_sec:.2f} seconds")
    
    # Adjust threshold based on audio length
    silence_threshold = 0.015
    if audio_duration_sec > 10:
        # For longer files, be more aggressive with silence removal
        silence_threshold = 0.02
    
    context_audio = remove_silence(context_audio, threshold=silence_threshold, 
                                  min_silence_duration=0.15, 
                                  sample_rate=generator.sample_rate)
    
    processed_duration_sec = len(context_audio) / generator.sample_rate
    print(f"Processed audio duration: {processed_duration_sec:.2f} seconds")
    
    # Use the entire audio file for better voice cloning
    print(f"Using the entire processed audio file ({processed_duration_sec:.2f} seconds) for voice cloning")
    
    # Create context segment
    context_segments = [
        Segment(
            text=context_text,
            speaker=speaker_id,
            audio=context_audio
        )
    ]
    
    # Preprocess text for better pronunciation
    # Add punctuation if missing to help with phrasing
    if not any(p in text for p in ['.', ',', '!', '?']):
        text = text + '.'
        
    conversation_history = []
    conversation_history.append({"role": "assistant", "content": context_text})
    while True:
        try:
            pwii = []
            text = input("Enter text: ")
            if (text == "$CLEAR$"):
                print("Clearing context...")
                context_segments = [
                    Segment(text=context_text, speaker=speaker_id, audio=context_audio)
                ]
                conversation_history.append({"role": "assistant", "content": context_text})
                print("Cleared.")
            elif (text == "$SWAP$"):
                spkr += 1
            elif (text == "$BACK$"):
                spkr -= 1
            else:
                if ("||" in text):
                    pwii = text.split('||')
                print(f"Generating audio for: '{text if not pwii else pwii[0]}'")
                
                # Generate audio with context
                audio = generator.generate(
                    text=text if not pwii else pwii[0],
                    speaker=speaker_id,
                    context=context_segments,
                    max_audio_length_ms=25_000,  # Adjusted based on your feedback
                    temperature=0.6,  # Lower temperature for more accurate pronunciation
                    topk=20,  # More focused sampling for clearer speech
                    max_seq_len=4096 # Increase max seq len for longer audio but might cause issues so you might need to edit in the models.py as well 
                )
                
                
                # Save the audio to the output directory
                torchaudio.save(output_filename, audio.unsqueeze(0).cpu(), generator.sample_rate)
                print(f"Speech with context generated successfully! Saved to {output_filename}")
                context_segments.append(
                    Segment(text=text if not pwii else pwii[0], speaker=speaker_id, audio=audio)
                )
                
                if len(context_segments) > 5:
                    context_segments = context_segments[-5:]
                        
                conversation_history.append({"role": "user", "content": text if not pwii else pwii[0]})
                
                if (pwii):
                    print(f"Generating audio for: '{pwii[1]}'")
                
                    audio = generator.generate(
                        text=pwii[1],
                        speaker=spkr,
                        context=context_segments,
                        max_audio_length_ms=25_000,
                        max_seq_len=4096
                    )
                    
                    print("Saving audio to secondary.wav...")
                    torchaudio.save("secondary.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
                    print("Audio saved successfully!")
                    context_segments.append(
                        Segment(text=pwii[1], speaker=spkr, audio=audio)
                    )
                    
                    if len(context_segments) > 5:
                        context_segments = context_segments[-5:]
                        
                    conversation_history.append({"role": "user", "content": pwii[1]})
                    print("Fusing audio...")
                    run(shlex.split(f"ffmpeg -i {output_filename} -i secondary.wav -filter_complex '[0:0][1:0]concat=n=2:v=0:a=1[out]' -map '[out]' outputCombined.wav"))
            
        except Exception as e:
            print(f"Error generating audio: {e}")
            traceback.print_exc()
    return

def main():
    
    context_audio_path = "example.wav"

    context_text = "This is some text that can be captioned by a model like whisper.cpp or openai whisper."
    
    # Generate speech using the approach that worked best
    print("Generating speech with the successful approach...")
    generate_speech_with_context(
        text="Hello, this is my voice speaking with context.",
        speaker_id=999,
        context_audio_path=context_audio_path,
        context_text=context_text,
        output_filename="cloned_voice.wav"
    )
    
if __name__ == "__main__":
    main()
