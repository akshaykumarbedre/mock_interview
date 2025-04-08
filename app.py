import os
import json
import pyttsx3
import speech_recognition as sr

from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Initialize recognizer and mic
recognizer = sr.Recognizer()
mic = sr.Microphone()

# LLM setup
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Prompt template
feedback_template = """
You are a helpful visa interview coach. Your feedback will be spoken aloud to the applicant through text-to-speech, so write in a natural speaking style without sections, bullet points, or special formatting.

QUESTION ASKED: {question}
APPLICANT'S ANSWER: {answer}
NEXT QUESTION IN QUEUE: {followup_question}

Provide brief, conversational feedback (60-80 words) that:
1. Acknowledges one strength in their answer
2. Suggests one specific improvement
3. Smoothly transitions to the next question

Guidelines:
- Use natural speech patterns suitable for speaking aloud
- Avoid phrases like "FEEDBACK:", "STRENGTHS:", or any headings
- Don't repeat the applicant's full answer back to them
- Don't use special characters, bullet points, or numbering
- End by clearly stating "Now for your next question:" followed by the exact text of the follow-up question, if there is no follow-up question conclude it key point to improve 
"""

feedback_prompt = PromptTemplate(
    input_variables=["question", "answer", "followup_question"],
    template=feedback_template
)
chain = feedback_prompt | llm

questions = [
    "Why did you choose this particular time to travel?",
    "What is the purpose of getting visa?",
    "Why are you traveling at this time?",
    "Have you traveled abroad before?",
    "Do you have travel insurance?",
    "What do you do for a living in your home country?"
]

def clean_text(text):
    """Removes extra spaces, newlines, etc., so it's spoken in one smooth line."""
    return " ".join(text.split())

def speak(text):
    line = clean_text(text)
    engine.say(line)
    engine.runAndWait()

def listen_to_answer():
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print("🎙️ Listening...")
        audio = recognizer.listen(source)

    try:
        response = recognizer.recognize_google(audio)
        print(f"🗣️ You said: {response}")
        return response
    except sr.UnknownValueError:
        print("❌ Sorry, I couldn't understand you.")
        return "[Unrecognized speech]"
    except sr.RequestError:
        print("⚠️ Could not reach the speech service.")
        return "[Speech service error]"

def run_mock_interview():
    responses = []
    print("🤖 Starting your mock interview...")
    speak("Welcome to your mock interview. Let's begin.")
    speak(questions[0])

    for i, q in enumerate(questions):
        print(f"\n🧠 Question: {q}")

        answer = listen_to_answer()
        next_question = questions[i + 1] if i + 1 < len(questions) else "No further questions."

        result = chain.invoke({
            'question': q,
            'answer': answer,
            'followup_question': next_question
        })

        speak(result.content)

        responses.append({
            "question": q,
            "answer": answer,
            "feedback": result.content
        })

        # input("Press Enter for the next question...")

    with open("interview_responses.json", "w", encoding='utf-8') as f:
        json.dump(responses, f, indent=4)

    speak("Thank you. This concludes your mock interview.")
    print("\n✅ Done! Your responses have been saved to 'interview_responses.json'.")

# Start the mock interview
run_mock_interview()
