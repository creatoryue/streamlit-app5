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

WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={
        "video": False,
        "audio": True,
    },
)


def main():
    
    st.header("# Classificaion for lung condition demo.")
    "### Recording"
    
    webrtc_ctx = webrtc_streamer(
        key="sendonly-audio",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1536, #256 = 5 seconds
        client_settings=WEBRTC_CLIENT_SETTINGS,
    )
    
    if not webrtc_ctx.audio_receiver:
        st.info('Now condition: Stop recording.')
    
    global audio_frames, sound_chunk
    
    @st.cache
    def sayHello():
        st.text('123')
        

        
    if webrtc_ctx.audio_receiver:
        st.info('Now strat recording.\n Please breathe toward the microphone.')
        try:
            audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
        except:
            logger.warning("Queue is empty. Abort.")
            st.error('ohoh')
            
        sound_chunk = pydub.AudioSegment.empty()
        for audio_frame in audio_frames:
            sound = pydub.AudioSegment(
                data=audio_frame.to_ndarray().tobytes(),
                sample_width=audio_frame.format.bytes,
                frame_rate=audio_frame.sample_rate,
                channels=len(audio_frame.layout.channels),
            )
            sound_chunk += sound    
    
    state_btn_save = st.button('Save')
    if state_btn_save:
        try:
            sound_chunk.export('C:\\D\\NTHU\\ProjactStudy\\Python\\stramlit_WebAppLungClassification\\github\\APP5\\temp.wav', format='wav')
            st.info("Writing wav to disk")
        except:
            st.error('Try do recording first and do saving.')
            
    state_button = st.button('Click to show the data')
    if state_button:
        try:
            # st.text('Click!')
            sound_chunk = sound_chunk.set_channels(1) # Stereo to mono
            sample = np.array(sound_chunk.get_array_of_samples())
            st.text(sample)
        except:
            st.error('Try do recording first and do saving.')

    # fig_place = st.empty()

    # fig, [ax_time, ax_freq] = plt.subplots(
    #     2, 1, gridspec_kw={"top": 1.5, "bottom": 0.2}
    # )
    
    # sound_window_len = 5000  # 5s
    # sound_window_buffer = None
    

    # while True:
    #     if webrtc_ctx.audio_receiver:
    #         try:
    #             audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
    #         except queue.Empty:
    #             logger.warning("Queue is empty. Abort.")
    #             break

    #         sound_chunk = pydub.AudioSegment.empty()
    #         for audio_frame in audio_frames:
    #             sound = pydub.AudioSegment(
    #                 data=audio_frame.to_ndarray().tobytes(),
    #                 sample_width=audio_frame.format.bytes,
    #                 frame_rate=audio_frame.sample_rate,
    #                 channels=len(audio_frame.layout.channels),
    #             )
    #             sound_chunk += sound

    #         if len(sound_chunk) > 0:
    #             if sound_window_buffer is None:
    #                 sound_window_buffer = pydub.AudioSegment.silent(
    #                     duration=sound_window_len
    #                 )

    #             sound_window_buffer += sound_chunk
    #             if len(sound_window_buffer) > sound_window_len:
    #                 sound_window_buffer = sound_window_buffer[-sound_window_len:]

    #         if sound_window_buffer:
    #             # Ref: https://own-search-and-study.xyz/2017/10/27/python%E3%82%92%E4%BD%BF%E3%81%A3%E3%81%A6%E9%9F%B3%E5%A3%B0%E3%83%87%E3%83%BC%E3%82%BF%E3%81%8B%E3%82%89%E3%82%B9%E3%83%9A%E3%82%AF%E3%83%88%E3%83%AD%E3%82%B0%E3%83%A9%E3%83%A0%E3%82%92%E4%BD%9C/  # noqa
    #             sound_window_buffer = sound_window_buffer.set_channels(
    #                 1
    #             )  # Stereo to mono
    #             sample = np.array(sound_window_buffer.get_array_of_samples())

    #             ax_time.cla()
    #             times = (np.arange(-len(sample), 0)) / sound_window_buffer.frame_rate
    #             ax_time.plot(times, sample)
    #             ax_time.set_xlabel("Time")
    #             ax_time.set_ylabel("Magnitude")

    #             spec = np.fft.fft(sample)
    #             freq = np.fft.fftfreq(sample.shape[0], 1.0 / sound_chunk.frame_rate)
    #             freq = freq[: int(freq.shape[0] / 2)]
    #             spec = spec[: int(spec.shape[0] / 2)]
    #             spec[0] = spec[0] / 2

    #             ax_freq.cla()
    #             ax_freq.plot(freq, np.abs(spec))
    #             ax_freq.set_xlabel("Frequency")
    #             ax_freq.set_yscale("log")
    #             ax_freq.set_ylabel("Magnitude")

    #             fig_place.pyplot(fig)
    #     else:
    #         logger.warning("AudioReciver is not set. Abort.")
    #         break
    
    
    
    # st.info("Writing wav to disk")
    # sound_window_buffer.export('temp.wav', format='wav')    


    
    
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
