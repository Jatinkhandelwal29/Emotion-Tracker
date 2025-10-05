import streamlit as st
import requests
import numpy as np
import pandas as pd
import cv2
from datetime import datetime, timedelta
import plotly.express as px
import mediapipe as mp
from deepface import DeepFace

# --- Setup and Global Variables ---
BACKEND_URL = "http://localhost:5000"
MP_HANDS = mp.solutions.hands
MP_FACE_MESH = mp.solutions.face_mesh

# Dictionary of quotes based on dominant emotion
EMOTION_QUOTES = {
    "happy": "Happiness is not something readymade. It comes from your own actions. - Dalai Lama",
    "sad": "The sun shines not on us, but in us. - John Muir. Hold on, better times are ahead.",
    "anger": "Speak when you are angry and you will make the best speech you will ever regret. - Ambrose Bierce. Take a deep breath.",
    "fear": "The only thing we have to fear is fear itself. - Franklin D. Roosevelt. You are stronger than you think.",
    "disgust": "Disgust is a call for change. What is one small thing you can improve today?",
    "surprise": "Life is full of unexpected twists and turns. Embrace the wonder!",
    "neutral": "A calm mind is the ultimate weapon. You are at peace and present."
}

def get_quote(emotion):
    """Returns a quote based on the detected emotion."""
    # Use .lower() because DeepFace returns lowercase emotion strings
    return EMOTION_QUOTES.get(emotion.lower(), "Every emotion tells a story. Listen to yours carefully.")

# --- Authentication Pages ---
def login_page():
    """Renders the login form and handles API calls."""
    st.title("Login to Emotion Tracker")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    if st.button("Login"):
        if not username or not password:
            st.error("Please enter both username and password.")
            return

        try:
            response = requests.post(
                f"{BACKEND_URL}/login",
                json={"username": username, "password": password}
            )
            data = response.json()
            if response.status_code == 200:
                st.session_state['logged_in'] = True
                st.session_state['user_id'] = data.get('user_id')
                st.session_state['username'] = username
                st.success("Login successful!")
                st.rerun()
            else:
                st.error(data.get('message', 'Login failed.'))
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to backend: {e}")

def signup_page():
    """Renders the signup form and handles API calls."""
    st.title("Sign Up for Emotion Tracker")
    username = st.text_input("Username", key="signup_username")
    password = st.text_input("Password", type="password", key="signup_password")
    if st.button("Sign Up"):
        if not username or not password:
            st.error("Please enter both username and password.")
            return

        try:
            response = requests.post(
                f"{BACKEND_URL}/signup",
                json={"username": username, "password": password}
            )
            data = response.json()
            if response.status_code == 201:
                st.success("Account created successfully! You can now log in.")
                st.info("Please log in with your new credentials.")
            else:
                st.error(data.get('message', 'Signup failed.'))
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to backend: {e}")

# --- Webcam and Gesture/Emotion Recognition ---
def thumbs_up_detected(landmarks):
    """Checks for a thumbs-up gesture based on hand landmarks.
    This updated function is more robust and checks for the relative positions
    of multiple fingers to avoid false positives.
    """
    thumb_tip = landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
    thumb_ip = landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_IP]
    thumb_mcp = landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_MCP]
    
    index_tip = landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]
    
    middle_tip = landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_mcp = landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP]

    ring_tip = landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
    ring_mcp = landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_MCP]
    
    pinky_tip = landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP]
    pinky_mcp = landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_MCP]

    # Check if the thumb is up (above the rest of the hand)
    thumb_is_up = thumb_tip.y < thumb_ip.y and thumb_ip.y < thumb_mcp.y

    # Check if the thumb is pointing right (for a right hand) or left (for a left hand)
    hand_is_left = index_mcp.x > pinky_mcp.x
    if hand_is_left:
        # For a left hand, tip is to the right of mcp
        thumb_is_out = thumb_tip.x > thumb_mcp.x
    else:
        # For a right hand, tip is to the left of mcp
        thumb_is_out = thumb_tip.x < thumb_mcp.x
    
    # Check if the other fingers are bent or "down"
    fingers_are_down = (
        index_tip.y > index_mcp.y and
        middle_tip.y > middle_mcp.y and
        ring_tip.y > ring_mcp.y and
        pinky_tip.y > pinky_mcp.y
    )
    
    return thumb_is_up and fingers_are_down

# Callback to set the flag for taking a photo
def set_take_photo_flag():
    """Sets a session state flag when the 'Take Photo' button is clicked."""
    st.session_state.take_photo_flag = True

# --- Main App Logic ---
def main_app():
    """Main application page after successful login, now including photo capture and quotes."""
    st.sidebar.title(f"Welcome, {st.session_state.get('username')}")
    st.sidebar.write("User ID:", st.session_state.get('user_id'))
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.rerun()

    st.title("Emotion & Quote Tracker")
    st.markdown("""
    **Instructions:**
    1. Click **Start Camera** to begin live detection.
    2. Make a **Thumbs-Up** gesture to automatically save the current emotion.
    3. Click **Take Photo** to capture a frame without stopping the feed.
    """)
    st.warning("Please allow webcam access for emotion and gesture detection.")

    # --- Webcam Controls (Start, Stop, Take Photo) ---
    col_start, col_stop, col_photo = st.columns([1, 1, 1])
    
    # Start Camera button
    with col_start:
        if st.button("Start Camera", key="start_cam_btn"):
            st.session_state.is_running = True
            # Reset photo state when starting camera
            st.session_state.captured_image = None
    
    # Stop Camera button
    with col_stop:
        if st.button("Stop Camera", key="stop_cam_btn"):
            st.session_state.is_running = False
    
    # Take Photo button (calls the callback to set the flag)
    with col_photo:
        is_running = st.session_state.get('is_running', False)
        if st.button("Take Photo", key="take_photo_btn", on_click=set_take_photo_flag, disabled=not is_running):
            pass # Button only sets flag via callback

    # --- Placeholders for Live Feed and Info ---
    frame_placeholder = st.empty()
    quote_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Check if camera is running
    if st.session_state.is_running:
        # Initialize video capture (0 is usually the default webcam)
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            status_placeholder.error("Error: Could not open webcam. Please check if it's connected and accessible.")
            st.session_state.is_running = False
            return

        with MP_HANDS.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
            while st.session_state.is_running:
                ret, frame = cap.read()
                if not ret:
                    status_placeholder.error("Failed to capture frame from webcam.")
                    st.session_state.is_running = False
                    break
                
                # Flip the image horizontally for a natural view
                frame = cv2.flip(frame, 1)
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hands_results = hands.process(image_rgb)
                
                emotion_label = "Neutral"
                
                try:
                    # Emotion detection using DeepFace
                    analysis = DeepFace.analyze(
                        img_path=frame,
                        actions=['emotion'],
                        enforce_detection=False,
                        silent=True
                    )
                    
                    if len(analysis) > 0:
                        emotion_label = analysis[0]['dominant_emotion'].lower()
                        
                        # Draw bounding box and emotion label on the frame
                        face_info = analysis[0]['region']
                        x, y, w, h = face_info['x'], face_info['y'], face_info['w'], face_info['h']
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                        cv2.putText(frame, emotion_label.upper(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

                except Exception:
                    # DeepFace might fail if no face is detected
                    pass 

                # --- Quote Feature ---
                current_quote = get_quote(emotion_label)
                quote_placeholder.info(f"**Current Emotion: {emotion_label.upper()}**\n\n**Quote:** *{current_quote}*")
                
                # Hand gesture detection
                if hands_results.multi_hand_landmarks:
                    for hand_landmarks in hands_results.multi_hand_landmarks:
                        mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                        
                        if thumbs_up_detected(hand_landmarks):
                            status_placeholder.success("THUMBS UP! Saving emotion...")
                            
                            # Send emotion to backend
                            user_id = st.session_state.get('user_id')
                            if user_id:
                                try:
                                    payload = {
                                        "user_id": user_id,
                                        "username": st.session_state.get('username'),
                                        "emotion": emotion_label,
                                        "timestamp": datetime.now().isoformat()
                                    }
                                    requests.post(f"{BACKEND_URL}/emotion", json=payload)
                                except requests.exceptions.RequestException:
                                    st.sidebar.error("Failed to save emotion to backend.")
                            # Note: In a production Streamlit app, you might use a more robust
                            # threading approach or an interval check for debouncing the save.
                            
                # --- Take Photo Feature Logic ---
                if st.session_state.take_photo_flag:
                    # Capture the current frame (which is 'frame')
                    st.session_state.captured_image = frame.copy() 
                    st.session_state.take_photo_flag = False # Reset the flag immediately
                    status_placeholder.success("Photo captured successfully! View below.")
                    
                # Display the processed frame
                frame_placeholder.image(frame, channels="BGR", caption="Live Webcam Feed")
                
        # Release the camera when the loop ends
        cap.release()
        status_placeholder.info("Camera stopped.")

    # --- Display Captured Photo ---
    if st.session_state.get('captured_image') is not None:
        st.markdown("---")
        st.subheader("Captured Photo")
        # Display the BGR image saved from the loop
        st.image(st.session_state.captured_image, channels="BGR", caption=f"Photo captured at {datetime.now().strftime('%H:%M:%S')}")
        
    st.header("Emotion History & Analytics")
    
    user_id = st.session_state.get('user_id')
    if user_id:
        try:
            response = requests.get(f"{BACKEND_URL}/history/{user_id}")
            if response.status_code == 200:
                history_data = response.json()
                if not history_data:
                    st.info("No emotion data recorded yet.")
                    # return
                
                df = pd.DataFrame(history_data)
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
                    # --- Emotion Analytics ---
                    
                    st.subheader("Daily Emotion Timeline")
                    st.write("Emotion over the last 24 hours.")
                    
                    last_24_hours = datetime.now() - timedelta(hours=24)
                    daily_df = df[df['timestamp'] >= last_24_hours].sort_values(by='timestamp')
                    
                    if not daily_df.empty:
                        # Plot emotions over time
                        fig_timeline = px.line(daily_df, x='timestamp', y='emotion', markers=True, title='Daily Emotion Timeline')
                        fig_timeline.update_layout(yaxis_title="Emotion")
                        st.plotly_chart(fig_timeline)
                    else:
                        st.info("Not enough data in the last 24 hours for a timeline chart.")

                    st.subheader("Weekly Emotion Distribution")
                    st.write("Percentage of each emotion in the last 7 days.")
                    
                    last_7_days = datetime.now() - timedelta(days=7)
                    weekly_df = df[df['timestamp'] >= last_7_days]
                    
                    if not weekly_df.empty:
                        emotion_counts = weekly_df['emotion'].value_counts().reset_index()
                        emotion_counts.columns = ['emotion', 'count']
                        fig_pie = px.pie(emotion_counts, values='count', names='emotion', title='Weekly Emotion Distribution')
                        st.plotly_chart(fig_pie)
                    else:
                        st.info("Not enough data in the last 7 days for a pie chart.")
                    
                    st.subheader("Monthly Emotion Trend")
                    st.write("Change in emotion frequency over the last 30 days.")
                    
                    last_30_days = datetime.now() - timedelta(days=30)
                    monthly_df = df[df['timestamp'] >= last_30_days]
                    
                    if not monthly_df.empty:
                        monthly_df['day'] = monthly_df['timestamp'].dt.date
                        monthly_trends = monthly_df.groupby(['day', 'emotion']).size().reset_index(name='count')
                        fig_trend = px.line(monthly_trends, x='day', y='count', color='emotion', title='Monthly Emotion Trends')
                        st.plotly_chart(fig_trend)
                    else:
                        st.info("Not enough data in the last 30 days for a trend chart.")

                    # --- CSV Download ---
                    st.markdown("---")
                    st.subheader("Download Your History")
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        label="Download Emotion History as CSV",
                        data=csv_data,
                        file_name=f"emotion_history_{st.session_state.get('username')}.csv",
                        mime="text/csv"
                    )
                
            else:
                st.error("Failed to fetch emotion history.")
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to backend: {e}")

# --- Streamlit Session State Management ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'page' not in st.session_state:
    st.session_state['page'] = 'login'
# Initialize new session state variables for camera control
if 'is_running' not in st.session_state:
    st.session_state['is_running'] = False
if 'take_photo_flag' not in st.session_state:
    st.session_state['take_photo_flag'] = False
if 'captured_image' not in st.session_state:
    st.session_state['captured_image'] = None


# --- Page Navigation ---
st.sidebar.title("Navigation")
if st.sidebar.button("Login"):
    st.session_state['page'] = 'login'
if st.sidebar.button("Sign Up"):
    st.session_state['page'] = 'signup'

if st.session_state['logged_in']:
    main_app()
else:
    if st.session_state['page'] == 'login':
        login_page()
    elif st.session_state['page'] == 'signup':
        signup_page()
