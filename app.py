import streamlit as st
import tensorflow as tf
import pickle
import time
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- 1. Page Setup & Native Theme ---
st.set_page_config(page_title="Sarcasm Detector AI", page_icon="🎭", layout="centered")

# --- Clean, Safe CSS for Button & Headers ---
st.markdown("""
<style>
    /* Premium Button Styling */
    div.stButton > button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 8px;
        height: 55px;
        font-size: 20px;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease-in-out;
    }
    div.stButton > button:hover {
        background-color: #FF6B6B;
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(255, 75, 75, 0.4);
    }
    /* Centered Text */
    .title { text-align: center; font-size: 3rem; font-weight: 800; margin-bottom: 0px; }
    .subtitle { text-align: center; font-size: 1.2rem; color: #888; margin-bottom: 30px; }
</style>
""", unsafe_allow_html=True)

# --- 2. Load AI Model ---
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model('sarcasm_model.h5')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

try:
    model, tokenizer = load_artifacts()
except Exception:
    st.error("⚠️ AI Brain not found! Please run your Jupyter notebook first to generate the model files.")
    st.stop()

MAX_LENGTH, PADDING_TYPE, TRUNC_TYPE = 100, 'post', 'post'

# --- 3. UI Header ---
st.markdown("<div class='title'>🎭 Sarcasm Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Is it Real News or <i>The Onion</i>? Let the AI decide.</div>", unsafe_allow_html=True)

# --- 4. Interactive Input Section ---
st.markdown("### 📝 Enter a News Headline:")
user_input = st.text_area(
    label="Headline Input",
    placeholder="e.g., Man excited to work 60 hours a week for minimum wage...",
    height=120,
    label_visibility="collapsed"
)

st.write("") # Quick spacing

# --- 5. Predict Button ---
# use_container_width=True forces it to be perfectly aligned with the text box
if st.button("🔍 Analyze Tone", use_container_width=True):
    
    if not user_input.strip():
        st.toast("⚠️ Please enter a headline first!", icon="🚨")
    else:
        # --- Interactivity: Fake Loading Spinner ---
        with st.spinner("🧠 AI is analyzing the text..."):
            time.sleep(1.5) # Adds a 1.5 second delay so it feels like it's doing heavy work!
            
            # Preprocess & Predict
            seq = tokenizer.texts_to_sequences([user_input])
            padded = pad_sequences(seq, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNC_TYPE)
            prediction = model.predict(padded)[0][0]
            confidence = prediction * 100

        # --- 6. Beautiful Dashboard Results ---
        st.divider()
        st.markdown("<h3 style='text-align: center;'>📊 Analysis Results</h3>", unsafe_allow_html=True)
        st.write("")

        # Use columns to put the label and the score side-by-side
        col1, col2 = st.columns(2)
        
        if prediction > 0.5:
            st.toast("Sarcasm Detected!", icon="🙄") # Pop-up notification
            
            with col1:
                st.error("### 🙄 SARCASTIC")
                st.write("**Verdict:** The AI believes this is satirical, fake, or purely sarcastic.")
            with col2:
                # Dashboard Metric
                st.metric(label="AI Confidence Score", value=f"{confidence:.1f}%", delta="High Sarcasm", delta_color="inverse")
                
            st.progress(int(confidence))

        else:
            st.toast("Genuine News Detected!", icon="📰") # Pop-up notification
            st.balloons() # Fun interactive animation
            
            with col1:
                st.success("### 😐 GENUINE NEWS")
                st.write("**Verdict:** The AI believes this is a real, straightforward headline.")
            with col2:
                # Dashboard Metric
                st.metric(label="AI Confidence Score", value=f"{100 - confidence:.1f}%", delta="Genuine", delta_color="normal")
                
            st.progress(int(100 - confidence))

        # Reset button hint
        st.caption("Try typing another headline above to run a new analysis!")