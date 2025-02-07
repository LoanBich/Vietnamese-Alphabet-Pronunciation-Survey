import io
from pathlib import Path
 
import numpy as np
import streamlit as st
from audiorecorder import audiorecorder
 
from src.models import load_model, predict_score
from src.utils import (
    add_vertical_space,
    unique_audio_filename,
    unique_session_id,
    upload_file,
)
 
st.set_page_config("Data Collection", ":material/home:")
 
 
LESSONS = [
    {"title": "Letter E", "id": "E"},
    {"title": "Letter H", "id": "H"},
    {"title": "Letter i", "id": "I"},
    {"title": "Letter L", "id": "L"},
    {"title": "Letter N", "id": "N"},
    {"title": "Letter Ơ", "id": "Ơ"},
    {"title": "Letter U", "id": "U"},
    {"title": "Letter V", "id": "V"},
]
N_LESSONS = len(LESSONS)
 
 
def next_step():
    st.session_state["step"] += 1
 
 
@st.fragment
def show_greeting():
    st.subheader("Thank you!")
    add_vertical_space(1)
 
    st.write(
        "We need to collect data about pronunciation errors of hearing-impaired students for my graduation research. "
        "You will be participated in a pronunciation test with 8 letters in the Vietnamese alphabet."
    )
    st.write(
        "In each test, you will watch a video (**no sound**) explain on how to pronounce a letter."
    )
    st.write("Then, you will try to pronounce it and record the result.")
    st.write(
        "The system will automatically evaluate your pronounciation and give a score."
    )
    st.write(
        "**If your score is lower than 4, please help us review the video and try to record your pronounciation again (at least 3 times).**"
    )
    st.write("Rest assured that all collected data are private and confidential.")
 
    st.markdown("**Note:**")
    st.write(
        "- Press the start and help us record your pronunciation loud and clear in a quite environment."
    )
    left_col, right_col = st.columns(2, border=True)
    left_col.markdown("![guideline](app/static/mic.png)")
    right_col.markdown("![guideline](app/static/audio_recorder.png)")
 
    add_vertical_space(1)
    if st.button(label="Start", type="primary"):
        next_step()
        st.rerun()
 
 
@st.fragment
def show_lesson(lesson):
    lesson_title = lesson["title"]
    lesson_id = lesson["id"]
    lesson_video_file = Path(__file__).parent / "assets" / "videos" / f"{lesson_id}.mov"
 
    st.subheader(f"#{st.session_state['step']}/{N_LESSONS}. {lesson_title}")
    st.video(lesson_video_file)
 
    add_vertical_space(1)
 
    st.subheader("Evaluation")
    st.write(
        "Please watch the video as many times as you like and record your pronounciation loud and clear. The system will evaluate and give you a score."
    )
    st.write(
        "**If your score is lower than 4, please help us review the video and try to record your pronounciation again (at least 3 times).**"
    )
 
    audio = audiorecorder(
        "",  # "Ấn để bắt đầu ghi âm",
        "",  # "Ấn để dừng lại",
        key=lesson_id,
    )
 
    if len(audio) > 0:
        st.info(
            "You can check your record to see if it's loud and clear. If not, please record again.",
            icon="ℹ️",
        )
        st.audio(audio.export().read())
 
    if st.button(label="Evaluate", type="primary"):
        if len(audio) > 0:
            with st.spinner("Processing..."):
                score = None
                waveform = np.asarray(
                    audio.set_frame_rate(16000).get_array_of_samples()
                ).T.astype(np.float32)
 
                try:
                    score = predict_score(model, waveform, actual_label=lesson_id)
                except:
                    st.error(
                        "We can't process your speech. Please record again in a quite environment.",
                        icon="🚨",
                    )
 
                # upload to Dropbox
                audio_buffer = io.BytesIO()
                audio.export(audio_buffer, format="wav", parameters=["-ar", str(16000)])
                upload_file(
                    audio_buffer.getvalue(),
                    unique_audio_filename(
                        st.session_state["session_id"],
                        lesson_id,
                        score,
                    ),
                )
 
                if score is not None and score > 3.8:
                    st.info(
                        f"Your score: **{score:.1f}**. Your pronounciation is really good."
                    )
                if score is not None and score <= 3.8:
                    st.warning(
                        f"Your score: **{score:.1f}**. You need practice. Please review the video and help us record again (**at least 3 times**)!"
                    )
        else:
            st.error(
                "The system can't record your speech. Please check and record again!",
                icon="🚨",
            )
 
    add_vertical_space(2)
    st.divider()
    columns = st.columns((3, 1))
    with columns[1]:
        if st.button(label="Next test ➡️", type="secondary", use_container_width=True):
            next_step()
            st.rerun()
 
 
@st.fragment
def show_thankyou():
    st.subheader("Done! Thank you!")
    add_vertical_space(1)
    st.write("We are really appriciate your time and your help. Have a nice day!")
 
 
st.title("Data Collection")
 
if "step" not in st.session_state:
    st.session_state["step"] = 0
 
if "session_id" not in st.session_state:
    st.session_state["session_id"] = unique_session_id()
 
model = load_model()
 
with st.container(border=True):
    if st.session_state["step"] == 0:
        show_greeting()
    elif st.session_state["step"] <= N_LESSONS:
        lesson_idx = st.session_state["step"] - 1
        show_lesson(LESSONS[lesson_idx])
    else:
        show_thankyou()
 
