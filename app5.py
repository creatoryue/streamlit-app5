import streamlit as st
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events

stt_button = Button(label="Speak", width=100)

stt_button.js_on_event("button_click", CustomJS(code="""
    const context = new AudioContext()
    
    setupContext()
    
    asyn function setupContext() {
        const microphone = await getMicrophone()
        const source = context.createMediaStreamSource(microphone)
        source.connect(context.destination)
    }
    
    function getMicrophone() {
        return navigator.mediaDevices.getUserMedia({
            audio:{
                echoCancellation: flase,
                autoGainControl: false,
                noiseSuppression: false,
                latency: 0
            }
        })
    }
    
    """))

result = streamlit_bokeh_events(
    stt_button,
    events="setupContext",
    key="listen",
    refresh_on_update=False,
    override_height=75,
    debounce_time=0)


