from uuid import uuid4

import dropbox
import streamlit as st

dbx = dropbox.Dropbox(
    oauth2_refresh_token=st.secrets["dropbox_refresh_token"],
    app_key=st.secrets["dropbox_app_key"],
    app_secret=st.secrets["dropbox_app_secret"],
)


def add_vertical_space(num_lines: int = 1) -> None:
    for _ in range(num_lines):
        st.write("")


def upload_file(data, filename) -> None:
    dropbox_file_path = f"/Survey/{filename}"

    try:
        dbx.files_upload(data, dropbox_file_path)
    except dropbox.exceptions.ApiError as err:
        print("Dropbox API error", err)
        return None


def unique_audio_filename(session_id: str, lesson_id: str, score: float | None) -> str:
    id = uuid4().hex
    return f"{session_id}/{lesson_id}_{id}_{score}.wav"


def unique_session_id() -> str:
    id = uuid4().hex
    return id
