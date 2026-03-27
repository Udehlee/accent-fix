import gradio as gr
import requests
import os

BACKEND_URL = os.getenv("BACKEND_URL")


def process_audio(audio_file, engine):
    if audio_file is None:
        return (
            "Please upload an audio file first.",
            "", "", "", ""
        )

    try:
        with open(audio_file, "rb") as f:
            response = requests.post(
                f"{BACKEND_URL}/upload-audio",
                files={"file": f},
                params={"engine": engine},
                timeout=120
            )

        if response.status_code != 200:
            return (
                f"Error: {response.json().get('detail', 'Unknown error')}",
                "", "", "", ""
            )

        data = response.json()

        accent_info = (
            f"Accent:      {data['accent']}\n"
            f"Confidence:  {data['accent_confidence']:.0%}\n"
            f"Engine:      {data['engine_used']}"
        )

        original = data["original_transcript"]
        corrected = data["corrected_transcript"]

        highlights = data.get("highlights", [])
        if highlights:
            highlight_text = f"Total Corrections: {data['total_corrections']}\n"
            highlight_text += "─" * 40 + "\n\n"
            for i, h in enumerate(highlights, 1):
                highlight_text += (
                    f"{i}.  {h['original_word']}  →  {h['corrected_word']}\n"
                    f"    Confidence : {h['confidence']:.0%}\n"
                    f"    Reason     : {h['explanation']}\n\n"
                )
        else:
            highlight_text = "No corrections were needed."

        summary = data["summary"]

        return accent_info, original, corrected, highlight_text, summary

    except requests.exceptions.ConnectionError:
        return ("Cannot connect to backend. Make sure FastAPI is running on port 8000.", "", "", "", "")
    except Exception as e:
        return (f"Something went wrong: {str(e)}", "", "", "", "")


# Custom CSS
css = """
/* Center the whole app and limit max width */
.gradio-container {
    max-width: 720px !important;
    margin: 0 auto !important;
    padding: 16px !important;
    font-family: sans-serif !important;
}

/* Header */
.app-header {
    text-align: center;
    padding: 24px 0 8px 0;
}

/* Make all textboxes look clean */
textarea {
    font-size: 14px !important;
    line-height: 1.6 !important;
}

/* Submit button full width */
#submit-btn {
    width: 100% !important;
    margin-top: 8px !important;
}

/* Card style for each section */
.section-card {
    border-radius: 12px !important;
    padding: 16px !important;
    margin-bottom: 12px !important;
}

/* Mobile responsiveness */
@media (max-width: 600px) {
    .gradio-container {
        padding: 8px !important;
    }
    textarea {
        font-size: 13px !important;
    }
}
"""

# UI Layout 
with gr.Blocks(css=css, title="AccentFix") as demo:

    # Header
    gr.HTML("""
        <div class="app-header">
            <h1 style="font-size:2rem; margin-bottom:4px;">🎙️ AccentFix</h1>
            <p style="color:#666; font-size:0.95rem; margin:0;">
                Upload your audio. AccentFix detects your accent and corrects 
                transcription errors automatically.
            </p>
        </div>
    """)

    # Upload Section 
    with gr.Group():
        audio_input = gr.Audio(
            label="Upload Audio",
            type="filepath",
            sources=["upload"],
        )

        engine_select = gr.Dropdown(
            label="Transcription Engine",
            choices=["assemblyai", "whisper"],
            value="assemblyai",
        )

        submit_btn = gr.Button(
            value="Process Audio",
            variant="primary",
            elem_id="submit-btn"
        )

    gr.HTML("<hr style='border:none;border-top:1px solid #eee;margin:8px 0'>")

    accent_output = gr.Textbox(
        label="Accent Detection",
        lines=3,
        interactive=False,
        placeholder="Accent details will appear here..."
    )

    #  Transcripts 
    original_output = gr.Textbox(
        label="Original Transcript",
        lines=5,
        interactive=False,
        placeholder="Raw transcript will appear here..."
    )

    corrected_output = gr.Textbox(
        label="Corrected Transcript",
        lines=5,
        interactive=False,
        placeholder="Corrected transcript will appear here..."
    )


    highlights_output = gr.Textbox(
        label="Corrections Made",
        lines=6,
        interactive=False,
        placeholder="List of corrections will appear here..."
    )

    summary_output = gr.Textbox(
        label="Summary",
        lines=3,
        interactive=False,
        placeholder="Summary will appear here..."
    )


    gr.HTML("""
        <p style="text-align:center;color:#999;font-size:0.8rem;margin-top:16px;">
            Supported formats: mp3 · wav · m4a · webm · ogg · flac
        </p>
    """)

    # Wire Button
    submit_btn.click(
        fn=process_audio,
        inputs=[audio_input, engine_select],
        outputs=[
            accent_output,
            original_output,
            corrected_output,
            highlights_output,
            summary_output
        ]
    )


if __name__ == "__main__":
    demo.launch(
        server_port=7860,
        share=True
    )