# record.py — lance depuis Windows PowerShell, pas WSL
import sounddevice as sd
import soundfile as sf

duration = 5  # secondes
sr = 22050

print("🎤 Enregistrement...")
audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
sd.wait()
sf.write("test.wav", audio, sr)
print("✅ Sauvegardé test.wav")