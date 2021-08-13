from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
from aiortc.contrib.media import MediaRecorder


def main():
    def recorder_factory():
        return MediaRecorder("record.wav")

    webrtc_streamer(
        key="sendonly-audio",
        mode=WebRtcMode.SENDONLY,
        in_recorder_factory=recorder_factory,
        client_settings=ClientSettings(
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={
                "audio": True,
                "video": False,
            },
        ),
    )


if __name__ == "__main__":
    main()
