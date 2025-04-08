# Import required libraries
import os
import json
import time
import pyttsx3
import speech_recognition as sr
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load environment variables (for API keys)
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Initialize speech recognition
recognizer = sr.Recognizer()
mic = sr.Microphone()

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Chat history handler
def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///memory.db")

# Define feedback template
feedback_template = """
You are a helpful visa interview coach. Your feedback will be spoken aloud to the applicant through text-to-speech, so write in a natural speaking style without sections, bullet points, or special formatting.

Provide brief, conversational feedback (60-80 words) that:
1. Acknowledges one strength in their answer
2. Suggests one specific improvement
3. Smoothly transitions to the next question

Guidelines:
- Use natural speech patterns suitable for speaking aloud
- Avoid phrases like "FEEDBACK:", "STRENGTHS:", or any headings
- Don't repeat the applicant's full answer back to them
- Don't use special characters, bullet points, or numbering
- End by clearly stating "Now for your next question:" followed by the exact text of the follow-up question, if there is no followup question conclude it key point to imporve 

Example of good response style:
"That's a good point about your return ticket. You might want to mention your job commitments back home as well, as this shows ties to your country. Now for your next question: What is the purpose of your visit?"
"""

# Set up the prompt and runnable
feedback_prompt = ChatPromptTemplate.from_messages([
    ("system", feedback_template),
    MessagesPlaceholder(variable_name="history"),
    ("human", "Question: {question}\nAnswer: {answer}\nNext question: {followup_question}")
])

runnable = feedback_prompt | llm

runnable_with_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# Define interview questions
questions = [
    "Why did you choose this particular time to travel?",
    "What is the purpose of getting vica?",
    "Why are you traveling at this time?",
    "Have you traveled abroad before?",
    "Do you have travel insurance?",
    "What do you do for a living in your home country?"
]

# Text-to-speech function
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Speech recognition function
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

# Main interview function
def run_mock_interview(session_id="default"):
    responses = []
    print("🤖 Starting your mock interview...")
    speak("Welcome to your mock interview. Let's begin.")
    
    for i, q in enumerate(questions):
        print(f"\n🧠 Question: {q}")
        speak(q)
        
        # Listen to the candidate's answer
        answer = listen_to_answer()
        
        # Determine the follow-up question
        next_question = questions[i + 1] if i + 1 < len(questions) else "No further questions."
        
        # Pass the question, answer, and follow-up question to the LLM
        result = runnable_with_history.invoke(
            {
                'question': q, 
                "answer": answer, 
                "followup_question": next_question, 
                'input': "some_input_value"
            },
            config={"configurable": {"session_id": session_id}}
        )
        
        # Speak the LLM's response
        feedback = result.content
        print(f"🤖 Feedback: {feedback}")
        speak(feedback)
        
        # Save the response
        responses.append({
            "question": q,
            "answer": answer,
            "feedback": feedback
        })
        
        if i < len(questions) - 1:
            input("Press Enter for the next question...")

    # Save responses to a file
    with open("interview_responses.json", "w", encoding='utf-8') as f:
        json.dump(responses, f, indent=4)

    speak("Thank you. This concludes your mock interview.")
    print("\n✅ Done! Your responses have been saved to 'interview_responses.json'.")

# Entry point
if __name__ == "__main__":
    run_mock_interview("session_" + str(int(time.time())))
