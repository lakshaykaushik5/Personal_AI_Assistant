import speech_recognition as sr

# Initialize the recognizer
r = sr.Recognizer()

# --- Adjustable Parameters ---
# Seconds of non-speaking audio before a phrase is considered complete.
# Increase this if the script stops while you are still speaking.
pause_threshold = 2.5  # Default is 0.8. Try increasing to 1.0, 1.5, or higher if needed.

# Seconds of audio to listen for the phrase.
# This is the maximum duration the phrase can be.
phrase_time_limit = 10 # Keep this as before or adjust as needed for maximum phrase length

# Set the pause threshold for the recognizer instance
r.pause_threshold = pause_threshold

# Use the default microphone as the audio source
def get_text_from_speech():
    with sr.Microphone() as source:
        print("Adjusting for ambient noise... Please wait.")
        # Adjusting for ambient noise helps set a good energy_threshold
        r.adjust_for_ambient_noise(source, duration=1)
        print(f"Set minimum energy threshold to: {r.energy_threshold}")
        print(f"Set pause threshold to: {r.pause_threshold}")
        print("Say something!")

        try:
            # Listen for the first phrase and extract it into audio data
            # Use the adjusted phrase_time_limit and the set pause_threshold implicitly
            audio = r.listen(source, timeout=10, phrase_time_limit=phrase_time_limit)

            print("Recognizing...")
            # Recognize speech using Google Web Speech API
            text = r.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.WaitTimeoutError:
            print("No speech detected within the initial 5-second timeout.")
        except sr.UnknownValueError:
            print("Google Web Speech could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Web Speech service; {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


