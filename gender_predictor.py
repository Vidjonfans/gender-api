import librosa
import numpy as np
import soundfile as sf # soundfile का उपयोग बेहतर I/O के लिए

def extract_features_with_clarity(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None) # sr=None original sample rate रखता है
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return None, None, "Error loading audio file."

    # 1. पिच निकालना
    # fmin और fmax को अपनी जरूरत के हिसाब से एडजस्ट करें
    try:
        pitch, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C3'), fmax=librosa.note_to_hz('C6'))
        
        # अमान्य पिच मानों को हटा दें (nan)
        valid_pitch = pitch[~np.isnan(pitch)]
        
        if len(valid_pitch) == 0:
            avg_pitch = 0 # यदि कोई वैध पिच नहीं मिली
        else:
            avg_pitch = np.mean(valid_pitch)
            
    except Exception as e:
        print(f"Error extracting pitch: {e}")
        avg_pitch = 0 # यदि पिच निकालने में त्रुटि होती है
        
    # 2. सिग्नल-टू-नॉइज़ रेश्यो (SNR) का अनुमान लगाना
    # यह एक सरल SNR अनुमान है. अधिक सटीक तरीकों के लिए उन्नत DSP की आवश्यकता होती है।
    
    # छोटे, नॉन-ओवरलैपिंग फ्रेम्स में ऑडियो को विभाजित करें
    frame_length = int(0.02 * sr) # 20 ms फ्रेम
    hop_length = int(0.01 * sr) # 10 ms हॉप
    
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length).T
    
    # प्रत्येक फ्रेम की RMS ऊर्जा
    rms_energy = np.array([np.sqrt(np.mean(frame**2)) for frame in frames])
    
    # नॉइज़ का अनुमान (शुरुआती 0.5 सेकंड से, यदि उपलब्ध हो)
    noise_frames = frames[:int(0.5 * sr / hop_length)]
    if len(noise_frames) > 0:
        noise_energy = np.mean([np.sqrt(np.mean(frame**2)) for frame in noise_frames])
    else:
        noise_energy = 0.0001 # बहुत कम डिफ़ॉल्ट शोर यदि कोई नॉइज़ फ्रेम नहीं है

    # सिग्नल एनर्जी (आवाज वाले हिस्सों से) - एक सरल अनुमान
    # हम मान सकते हैं कि उच्चतम RMS ऊर्जा वाले फ्रेम सिग्नल वाले हैं
    if len(rms_energy) > 0:
        signal_energy = np.percentile(rms_energy, 90) # शीर्ष 10% ऊर्जा को सिग्नल मानें
    else:
        signal_energy = 0.0001
        
    # SNR की गणना (dB में)
    if noise_energy > 0:
        snr_db = 10 * np.log10(signal_energy / noise_energy)
    else:
        snr_db = float('inf') # यदि कोई शोर नहीं है तो अनंत

    # 3. कुल ऑडियो ऊर्जा
    total_energy = np.sum(y**2) / len(y) # प्रति सैंपल औसत ऊर्जा
    
    # 4. स्पष्टता की जाँच (थ्रेशोल्ड आधारित)
    # ये थ्रेशोल्ड अनुभवजन्य हैं और आपको उन्हें अपने डेटा के अनुसार ट्यून करना पड़ सकता है।
    
    min_snr_threshold = 5.0 # न्यूनतम SNR (dB में) - इससे कम पर आवाज़ स्पष्ट नहीं
    min_energy_threshold = 0.001 # न्यूनतम औसत ऊर्जा - इससे कम पर आवाज़ बहुत धीमी या मौन
    
    clarity_message = "voice clear"
    if snr_db < min_snr_threshold:
        clarity_message = "Voice not clear: High background noise"
    elif total_energy < min_energy_threshold:
        clarity_message = "Voice not clear: Too low volume or silence"
    
    return avg_pitch, clarity_message, snr_db, total_energy

def predict_gender_with_clarity_check(file_path):
    avg_pitch, clarity_message, snr_db, total_energy = extract_features_with_clarity(file_path)

    if clarity_message != "voice clear":
        return clarity_message, None
    
    # लिंग का अनुमान
    # ये थ्रेशोल्ड सामान्य हैं, लेकिन भाषाओं और व्यक्ति-विशेष के अनुसार भिन्न हो सकते हैं।
    if avg_pitch == 0: # यदि पिच नहीं मिली (संभवतः बहुत कम या कोई आवाज नहीं)
        return "Cannot determine gender: No clear pitch detected", None
    elif avg_pitch > 165: # महिलाओं की पिच आमतौर पर ज़्यादा होती है
        gender = "Female"
    else: # पुरुषों की पिच आमतौर पर कम होती है
        gender = "Male"
        
    return gender, avg_pitch

# --- उपयोग का उदाहरण ---
if __name__ == "__main__":
    # परीक्षण के लिए कुछ डमी ऑडियो फाइलें बनाएं (यदि आपके पास नहीं हैं)
    # चेतावनी: ये डमी फाइलें वास्तविक ऑडियो को सटीक रूप से प्रस्तुत नहीं करेंगी
    # आपको वास्तविक ऑडियो फाइलों का उपयोग करना होगा।

    # उदाहरण 1: स्पष्ट पुरुष आवाज़ (कम पिच)
    # आप इसे अपनी वास्तविक ऑडियो फ़ाइल से बदलें
    clear_male_audio_path = "path/to/your/clear_male_voice.wav" 
    # Example: you can generate a dummy file for testing. In a real scenario, use actual recordings.
    # from scipy.io.wavfile import write
    # sample_rate = 22050
    # duration = 3 # seconds
    # frequency = 120 # Hz for male voice
    # t = np.linspace(0., duration, int(sample_rate * duration))
    # data = 0.5 * np.sin(2. * np.pi * frequency * t)
    # write(clear_male_audio_path, sample_rate, data.astype(np.float32))

    # उदाहरण 2: स्पष्ट महिला आवाज़ (उच्च पिच)
    # आप इसे अपनी वास्तविक ऑडियो फ़ाइल से बदलें
    clear_female_audio_path = "path/to/your/clear_female_voice.wav"
    # Example:
    # frequency = 220 # Hz for female voice
    # data = 0.5 * np.sin(2. * np.pi * frequency * t)
    # write(clear_female_audio_path, sample_rate, data.astype(np.float32))

    # उदाहरण 3: शोर वाली आवाज़ (कम SNR)
    # आप इसे अपनी वास्तविक ऑडियो फ़ाइल से बदलें
    noisy_audio_path = "path/to/your/noisy_voice.wav"
    # Example:
    # noise = np.random.normal(0, 0.2, len(t))
    # data = 0.3 * np.sin(2. * np.pi * 150 * t) + noise
    # write(noisy_audio_path, sample_rate, data.astype(np.float32))

    # उदाहरण 4: बहुत धीमी आवाज़ (कम ऊर्जा)
    # आप इसे अपनी वास्तविक ऑडियो फ़ाइल से बदलें
    low_volume_audio_path = "path/to/your/low_volume_voice.wav"
    # Example:
    # data = 0.01 * np.sin(2. * np.pi * 180 * t)
    # write(low_volume_audio_path, sample_rate, data.astype(np.float32))

    print("--- Clear Male Voice Example ---")
    gender_result, pitch_val = predict_gender_with_clarity_check(clear_male_audio_path)
    if pitch_val is not None:
        print(f"Predicted Gender: {gender_result}, Average Pitch: {pitch_val:.2f} Hz")
    else:
        print(f"Message: {gender_result}")

    print("\n--- Clear Female Voice Example ---")
    gender_result, pitch_val = predict_gender_with_clarity_check(clear_female_audio_path)
    if pitch_val is not None:
        print(f"Predicted Gender: {gender_result}, Average Pitch: {pitch_val:.2f} Hz")
    else:
        print(f"Message: {gender_result}")

    print("\n--- Noisy Voice Example ---")
    gender_result, pitch_val = predict_gender_with_clarity_check(noisy_audio_path)
    if pitch_val is not None:
        print(f"Predicted Gender: {gender_result}, Average Pitch: {pitch_val:.2f} Hz")
    else:
        print(f"Message: {gender_result}")

    print("\n--- Low Volume Voice Example ---")
    gender_result, pitch_val = predict_gender_with_clarity_check(low_volume_audio_path)
    if pitch_val is not None:
        print(f"Predicted Gender: {gender_result}, Average Pitch: {pitch_val:.2f} Hz")
    else:
        print(f"Message: {gender_result}")
