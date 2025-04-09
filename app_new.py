import os
import json
import time
import pyttsx3
import streamlit as st
import speech_recognition as sr
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# App title and description
st.set_page_config(page_title="Visa Interview Coach", page_icon="🎙️")
st.title("Visa Interview Coach")
st.markdown("Practice your visa interview and get AI feedback on your responses.")

# Initialize session state for tracking interview progress
if 'current_question_index' not in st.session_state:
    st.session_state.current_question_index = 0
if 'responses' not in st.session_state:
    st.session_state.responses = []
if 'session_id' not in st.session_state:
    st.session_state.session_id = f"session_{int(time.time())}"
if 'interview_active' not in st.session_state:
    st.session_state.interview_active = False
if 'interview_complete' not in st.session_state:
    st.session_state.interview_complete = False

# Load environment variables (for API keys)
load_dotenv()

# API key input
api_key = st.sidebar.text_input("Enter your Google API Key", type="password")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

# Define interview questions
questions = [
    "Why did you choose this particular time to travel?",
    "What is the purpose of getting a visa?",
    "Why are you traveling at this time?",
    "Have you traveled abroad before?",
    "Do you have travel insurance?",
    "What do you do for a living in your home country?"
]

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
- End by clearly stating "Now for your next question:" followed by the exact text of the follow-up question, if there is no followup question conclude it key point to improve 

Example of good response style:
"That's a good point about your return ticket. You might want to mention your job commitments back home as well, as this shows ties to your country. Now for your next question: What is the purpose of your visit?"
"""

# Set up the prompt and runnable
feedback_prompt = ChatPromptTemplate.from_messages([
    ("system", feedback_template),
    MessagesPlaceholder(variable_name="history"),
    ("human", "Question: {question}\nAnswer: {answer}\nNext question: {followup_question}")
])

# Initialize LLM (when API key is provided)
def get_llm_feedback(question, answer, followup_question):
    if not os.environ.get("GOOGLE_API_KEY"):
        return "Please provide a Google API key in the sidebar to get AI feedback."
    
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        runnable = feedback_prompt | llm
        runnable_with_history = RunnableWithMessageHistory(
            runnable,
            get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        
        result = runnable_with_history.invoke(
            {
                'question': question, 
                "answer": answer, 
                "followup_question": followup_question, 
                'input': "some_input_value"
            },
            config={"configurable": {"session_id": st.session_state.session_id}}
        )
        
        return result.content
    except Exception as e:
        st.error(f"Error getting feedback: {str(e)}")
        return f"Sorry, there was an error generating feedback: {str(e)}"

# Text-to-speech function using pyttsx3
def text_to_speech(text):
    # In a Streamlit app, we can't directly use pyttsx3 as it requires a GUI
    # Instead, we'll save the audio to a file and play it back
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.save_to_file(text, 'temp_speech.mp3')
        engine.runAndWait()
        
        # Play the audio file
        audio_file = open('temp_speech.mp3', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/mp3')
        audio_file.close()
    except Exception as e:
        st.warning(f"Could not generate speech: {str(e)}")

# Speech recognition function
def speech_to_text():
    # Create a placeholder for status messages
    status_placeholder = st.empty()
    status_placeholder.info("Initializing microphone...")
    
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            status_placeholder.info("Adjusting for ambient noise... Please wait.")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            status_placeholder.info("🎙️ Listening... Speak now!")
            audio = recognizer.listen(source, timeout=10)
        
        status_placeholder.info("Processing your speech...")
        response = recognizer.recognize_google(audio)
        status_placeholder.success("Speech recognized!")
        return response
    except sr.UnknownValueError:
        status_placeholder.error("Sorry, I couldn't understand what you said.")
        return ""
    except sr.RequestError:
        status_placeholder.error("Could not request results from speech recognition service.")
        return ""
    except Exception as e:
        status_placeholder.error(f"Error during speech recognition: {str(e)}")
        return ""

# Function to start the interview
def start_interview():
    st.session_state.interview_active = True
    st.session_state.current_question_index = 0
    st.session_state.responses = []
    st.session_state.interview_complete = False
    st.session_state.session_id = f"session_{int(time.time())}"

# Function to handle next question
def next_question():
    if st.session_state.current_question_index < len(questions) - 1:
        st.session_state.current_question_index += 1
    else:
        st.session_state.interview_complete = True

# Function to save responses to file
def save_responses():
    with open("interview_responses.json", "w", encoding='utf-8') as f:
        json.dump(st.session_state.responses, f, indent=4)
    return "interview_responses.json"

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    if not st.session_state.interview_active:
        if st.button("Start Interview", use_container_width=True):
            start_interview()
    
    if st.session_state.interview_complete:
        if st.button("Download Responses", use_container_width=True):
            filename = save_responses()
            st.success(f"Responses saved to {filename}")
        
        if st.button("Start New Interview", use_container_width=True):
            start_interview()

# Main interview UI
if st.session_state.interview_active:
    if not st.session_state.interview_complete:
        # Display current question
        current_q = questions[st.session_state.current_question_index]
        st.subheader(f"Question {st.session_state.current_question_index + 1} of {len(questions)}")
        st.markdown(f"### {current_q}")
        
        # Option to play question audio
        if st.button("🔊 Listen to Question"):
            text_to_speech(current_q)
        
        # User can choose to type or speak answer
        input_method = st.radio("Choose how to answer:", ("Type", "Speak"))
        
        if input_method == "Type":
            user_answer = st.text_area("Your answer:", height=150)
            submit_button = st.button("Submit Answer")
        else:
            user_answer = ""
            if st.button("🎙️ Start Speaking"):
                user_answer = speech_to_text()
                if user_answer:
                    st.success("Your answer was recorded!")
                    st.write(f"**Your answer:** {user_answer}")
            submit_button = st.button("Submit Spoken Answer") if user_answer else False
        
        # When answer is submitted
        if submit_button and user_answer:
            # Determine the follow-up question
            next_q = questions[st.session_state.current_question_index + 1] if st.session_state.current_question_index < len(questions) - 1 else "No further questions."
            
            # Get feedback
            with st.spinner("Getting feedback on your answer..."):
                feedback = get_llm_feedback(current_q, user_answer, next_q)
            
            # Display feedback
            st.markdown("### Feedback")
            st.write(feedback)
            
            # Play feedback audio
            if st.button("🔊 Listen to Feedback"):
                text_to_speech(feedback)
            
            # Save the response
            st.session_state.responses.append({
                "question": current_q,
                "answer": user_answer,
                "feedback": feedback
            })
            
            # Button to continue to next question
            if st.button("Next Question"):
                next_question()
                st.experimental_rerun()
    
    else:
        # Interview complete
        st.success("🎉 Congratulations! You've completed the mock visa interview.")
        st.subheader("Interview Summary")
        
        # Display all questions and answers
        for i, response in enumerate(st.session_state.responses):
            with st.expander(f"Question {i+1}: {response['question']}"):
                st.write("**Your answer:**")
                st.write(response["answer"])
                st.write("**Feedback:**")
                st.write(response["feedback"])
        
        # Save responses
        if st.button("Download Responses"):
            filename = save_responses()
            st.success(f"Responses saved to {filename}")
        
        # Option to start a new interview
        if st.button("Start New Interview"):
            start_interview()
            st.experimental_rerun()

else:
    # Welcome screen
    st.markdown("""
    ## Welcome to the Visa Interview Coach!
    
    This application will help you practice for your visa interview by:
    
    1. Asking you common visa interview questions
    2. Recording your spoken responses or allowing you to type them
    3. Providing AI-powered feedback on your answers
    4. Saving all your responses for later review
    
    To get started, click the "Start Interview" button in the sidebar.
    
    ### Requirements:
    - A microphone for speech input (optional)
    - Speakers for audio output (optional)
    - A Google API key (enter in the sidebar)
    """)

# Show progress bar
if st.session_state.interview_active and not st.session_state.interview_complete:
    progress = (st.session_state.current_question_index) / len(questions)
    st.progress(progress)