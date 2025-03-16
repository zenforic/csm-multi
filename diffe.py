from huggingface_hub import hf_hub_download
from generator import load_csm_1b, Segment
import torchaudio
import traceback
import os
import shlex
from subprocess import run

try:
    print("Loading model...")
    model_path = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt")
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    generator = load_csm_1b(model_path, device)
    print("Model loaded successfully!")
    reference_path = "reference.wav"
    context_segments = []
    refs = False
    spkr = 0
    
    if os.path.exists(reference_path):
        print(f"Using {reference_path} as reference audio")
        refs = True

        def load_audio(audio_path):
            audio_tensor, sample_rate = torchaudio.load(audio_path)
            audio_tensor = torchaudio.functional.resample(
                audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=generator.sample_rate
            )
            return audio_tensor
        reference_audio = load_audio(reference_path)
        reference_text = "This is Sesame. I say hi. And pwee. And more!"
        context_segments = [
            Segment(text=reference_text, speaker=0, audio=reference_audio)
        ]
        
        print("Reference audio loaded and added to context")
    else:
        print("No reference audio.wav found, starting without context")

    conversation_history = []
    if context_segments:
        conversation_history.append({"role": "assistant", "content": reference_text})

    while True:
        try:
            text = input("Enter text: ")
            if (text == "$CLEAR$"):
                print("Clearing context...")
                if refs:
                    context_segments = [
                        Segment(text=reference_text, speaker=0, audio=reference_audio)
                    ]
                    conversation_history.append({"role": "assistant", "content": reference_text})
                else:
                    conversation_history = []
                    context_segments = []
                print("Cleared.")
            elif (text == "$SWAP$"):
                spkr += 1
            elif (text == "$BACK$"):
                spkr -= 1
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
                )
                
                print("Saving audio to audio.wav...")
                torchaudio.save("audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
                print("Audio saved successfully!")
                context_segments.append(
                    Segment(text=text if not pwii else pwii[0], speaker=spkr, audio=audio)
                )
                
                if len(context_segments) > 5:
                    context_segments = context_segments[-5:]
                    
                conversation_history.append({"role": "user", "content": text if not pwii else pwii[0]})
                
                if (pwii):
                    generated = 0
                    for i in range(1, len(pwii)):
                        if (i > 4):
                            print("Too many segments, 4 max, skipping the rest.")
                            break
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
                            context=context_segments,
                            max_audio_length_ms=25_000,
                        )
                        
                        match (i):
                            case 1:
                                print("Saving audio to secondary.wav...")
                                torchaudio.save("secondary.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
                                print("Audio saved successfully!")
                            case 2:
                                print("Saving audio to secondary2.wav...")
                                torchaudio.save("secondary2.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
                                print("Audio saved successfully!")
                            case 3:
                                print("Saving audio to secondary3.wav...")
                                torchaudio.save("secondary3.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
                                print("Audio saved successfully!")
                            case 4:
                                print("Saving audio to secondary4.wav...")
                                torchaudio.save("secondary4.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
                                print("Audio saved successfully!")
                        
                        context_segments.append(
                            Segment(text=pwii[i], speaker=spkr + spkrOffset, audio=audio)
                        )
                        
                        if len(context_segments) > 5:
                            context_segments = context_segments[-5:]
                            
                        conversation_history.append({"role": "user", "content": pwii[i]})
                        generated += 1
                    print("Fusing audio...")
                    match (generated):
                        case 1:
                            run(shlex.split("ffmpeg -i audio.wav -i secondary.wav -filter_complex '[0:0][1:0]concat=n=2:v=0:a=1[out]' -map '[out]' outputCombined.wav"))
                        case 2:
                            run(shlex.split("ffmpeg -i audio.wav -i secondary.wav -i secondary2.wav -filter_complex '[0:0][1:0][2:0]concat=n=3:v=0:a=1[out]' -map '[out]' outputCombined.wav"))
                        case 3:
                            run(shlex.split("ffmpeg -i audio.wav -i secondary.wav -i secondary2.wav -i secondary3.wav -filter_complex '[0:0][1:0][2:0][3:0]concat=n=4:v=0:a=1[out]' -map '[out]' outputCombined.wav"))
                        case 4:
                            run(shlex.split("ffmpeg -i audio.wav -i secondary.wav -i secondary2.wav -i secondary3.wav -i secondary4.wav -filter_complex '[0:0][1:0][2:0][3:0][4:0]concat=n=5:v=0:a=1[out]' -map '[out]' outputCombined.wav"))
            
        except Exception as e:
            print(f"Error generating audio: {e}")
            traceback.print_exc()
except Exception as e:
    print(f"Error initializing model: {e}")
    traceback.print_exc() 
