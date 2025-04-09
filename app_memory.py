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
You are an expert AI engineering interview coach. Your feedback will be spoken aloud to the candidate through text-to-speech, so write in a natural speaking style without sections, bullet points, or special formatting.

Provide brief, conversational feedback (60-80 words) that:
1. Acknowledges one strength in their answer
2. Suggests one specific improvement
3. Smoothly transitions to the next question

Guidelines:
- Use natural speech patterns suitable for speaking aloud
- Avoid phrases like "FEEDBACK:", "STRENGTHS:", or any headings
- Don't repeat the candidate's full answer back to them
- Don't use special characters, bullet points, or numbering
- End by clearly stating "Now for your next question:" followed by the exact text of the follow-up question, if there is no followup question conclude with a key point to improve

Example of good response style:
"I like how you explained your experience with model deployment. Consider going into more detail about the specific metrics you used to evaluate performance. Now for your next question: How would you handle imbalanced data in a classification problem?"
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

questions = [
    "Tell me about a challenging machine learning project you've worked on.",
    "How do you approach feature selection in your ML models?",
    # "Explain how you would implement a recommendation system from scratch.",
    "What's your experience with deploying ML models to production?",
    "How do you handle missing data in a dataset?",
    "Explain the difference between supervised and unsupervised learning with examples.",
    # "How would you detect and handle outliers in your dataset?",
    "What metrics would you use to evaluate a classification model?",
    # "Describe your experience with deep learning frameworks.",
    # "How do you approach A/B testing for model improvements?"
]


# Text-to-speech function
def speak(text):
    engine.say(text)
    engine.runAndWait()

# def listen_to_answer():
#     with mic as source:
#         recognizer.adjust_for_ambient_noise(source)
#         print("🎙️ Listening... (Press Enter when done speaking)")
#         print("Start speaking now...")
        
#         # Set a longer phrase_time_limit (e.g., 60 seconds)
#         audio = recognizer.listen(source, phrase_time_limit=60,timeout=60)
        
#         # Optional: Allow manual control by pressing Enter to stop recording
#         # This would require a threading approach, which is more complex

#     try:
#         response = recognizer.recognize_google(audio)
#         print(f"🗣️ You said: {response}")
#         return response
#     except sr.UnknownValueError:
#         print("❌ Sorry, I couldn't understand you.")
#         return "[Unrecognized speech]"
#     except sr.RequestError:
#         print("⚠️ Could not reach the speech service.")
#         return "[Speech service error]"

# def listen_to_answer(duration=60):  # Record for 60 seconds
#     print(f"🎙️ You will have {duration} seconds to answer.")
#     print("Press Enter when ready to start...")
#     input()
    
#     with mic as source:
#         recognizer.adjust_for_ambient_noise(source)
#         print(f"Recording for {duration} seconds. Start speaking now...")
        
#         # Record for a fixed duration
#         audio = recognizer.record(source, duration=duration)
    
#     try:
#         response = recognizer.recognize_google(audio)
#         print(f"🗣️ You said: {response}")
#         return response
#     except sr.UnknownValueError:
#         print("❌ Sorry, I couldn't understand you.")
#         return "[Unrecognized speech]"
#     except sr.RequestError:
#         print("⚠️ Could not reach the speech service.")
#         return "[Speech service error]"

def listen_to_answer():
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print("🎙️ Listening... Recording will stop after 3 seconds of silence.")
        
        # Lower the pause threshold to 3 seconds instead of 5
        recognizer.pause_threshold = 2
        
        # Make the recognizer more responsive with these settings
        recognizer.dynamic_energy_threshold = True
       
       # May need adjustment based on your microphone
        
        # Listen with shorter phrase_time_limit to process speech more quickly
        audio = recognizer.listen(source, phrase_time_limit=180)

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
    speak(questions[0])
    
    for i, q in enumerate(questions):
        print(f"\n🧠 Question: {q}")
        # speak(q)
        
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
