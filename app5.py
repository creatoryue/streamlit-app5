import logging
from pathlib import Path
import streamlit as st
import pydub
import numpy as np
import queue
import matplotlib.pyplot as plt


from aiortc.contrib.media import MediaRecorder

from streamlit_webrtc import (
    AudioProcessorBase,
    ClientSettings,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)


HERE = Path(__file__).parent
logger = logging.getLogger(__name__)




def main():
    
    st.header("# Classificaion for lung condition demo.")
    "### Recording"
    
    sound_window_len = 5000  # 5s
    sound_window_buffer = None
    while True:
        if webrtc_ctx.audio_receiver:
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                logger.warning("Queue is empty. Abort.")
                break

            sound_chunk = pydub.AudioSegment.empty()
            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                sound_chunk += sound

            if len(sound_chunk) > 0:
                if sound_window_buffer is None:
                    sound_window_buffer = pydub.AudioSegment.silent(
                        duration=sound_window_len
                    )

                sound_window_buffer += sound_chunk
                if len(sound_window_buffer) > sound_window_len:
                    sound_window_buffer = sound_window_buffer[-sound_window_len:]
        else:
            logger.warning("AudioReciver is not set. Abort.")
            break
            
    st.info("Writing wav to disk")
    sound_window_buffer.export('temp.wav', format='wav')
    
    
    # sdata = app_sendonly_audio()
    
    # state_playButton = st.button("Click to show")
    # if state_playButton:
        # # st.text(sdata)
        # st.text('Hello!')
        # try:
        #     st.pyplot(sdata)
        # except:
        #    logger.warning("Error in plotting sdata.")
    #def recorder_factory():
    #    return MediaRecorder("record.wav")

    #webrtc_streamer(
    #    key="sendonly-audio",
    #    mode=WebRtcMode.SENDONLY,
    #    in_recorder_factory=recorder_factory,
    #    client_settings=ClientSettings(
     #       rtc_configuration={
    #            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    #        },
    #        media_stream_constraints={
    #            "audio": True,
    #            "video": False,
    #        },
    #    ),
    #)
    
    
    
if __name__ == '__main__':
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)
    
    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)
    
    main()
