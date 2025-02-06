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
    {"title": "Chữ cái E", "id": "E"},
    {"title": "Chữ cái H", "id": "H"},
    {"title": "Chữ cái i", "id": "I"},
    {"title": "Chữ cái L", "id": "L"},
    {"title": "Chữ cái N", "id": "N"},
    {"title": "Chữ cái Ơ", "id": "Ơ"},
    {"title": "Chữ cái U", "id": "U"},
    {"title": "Chữ cái V", "id": "V"},
]
N_LESSONS = len(LESSONS)


def next_step():
    st.session_state["step"] += 1


@st.fragment
def show_greeting():
    st.subheader("Thank you!")
    add_vertical_space(1)

    st.write(
        "Tớ đang cần thu thập data giọng nói về 8 chữ cái trong bảng chữ cái tiếng Việt để làm nghiên cứu cá nhân, hãy giúp tớ nhé!"
    )
    st.write(
        "Yên tâm là mọi dữ liệu tớ sẽ bảo mật và không có gì là nguy hiểm hết đâu ạ."
    )
    st.write(
        "Hãy ấn bắt đầu và giúp tớ thu âm các chữ cái, giúp tớ đọc chính xác rõ ràng từng chữ nhaaaa"
    )
    st.markdown("**Note:**")
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
    st.markdown("Sau khi học xong bài học, hãy đọc lại chữ cái đó để hệ thống chấm điểm")
    st.markdown(f"Hãy chỉ đọc chữ cái được yêu cầu. Ví dụ: **{lesson_id}**")

    audio = audiorecorder(
        "",  # "Ấn để bắt đầu ghi âm",
        "",  # "Ấn để dừng lại",
        key=lesson_id,
    )

    if len(audio) > 0:
        st.info(
            "Hãy giúp tớ check lại phát âm xem đã đúng và rõ ràng chưa nhé ạ! Nếu chưa được thì hãy ghi âm lại giúp tớ nha ạ!!",
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
                    if score > 3.8:
                        st.info("Bạn phát âm rất tốt!")
                    else:
                        st.warning(
                            "Bạn cần cải thiện thêm. Xem lại video và phát âm lại nhé!"
                        )
                except:
                    st.error(
                        "Giúp tớ thu âm lại nha, bạn nhớ phát âm to rõ nhé", icon="🚨"
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
        else:
            st.error("Chưa được rồi, giúp tớ thu âm lại nha", icon="🚨")

    add_vertical_space(2)
    if st.button(label="Âm tiếp theo", type="primary"):
        next_step()
        st.rerun()


@st.fragment
def show_thankyou():
    st.subheader("Done!")
    add_vertical_space(1)
    st.write("Thank you very much for your time and your help. Have a nice day!")


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
