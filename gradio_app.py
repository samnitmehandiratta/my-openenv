import gradio as gr
import requests
import json


# Test the API
def test_api():
    try:
        resp = requests.post("http://localhost:7860/reset", json={})
        return resp.status_code, resp.json()
    except Exception as e:
        return 0, str(e)


def reset_and_show():
    code, data = test_api()
    return f"Status: {code}\n\n{json.dumps(data, indent=2)[:500]}"


with gr.Blocks(title="Physics Surrogate") as demo:
    gr.Markdown("# Physics Surrogate Environment")
    gr.Markdown("OpenEnv API is running at POST /reset and POST /step endpoints")
    gr.Markdown("### Test API")
    btn = gr.Button("Test /reset")
    output = gr.Textbox(label="API Response")
    btn.click(fn=reset_and_show, outputs=output)
    gr.Markdown("This environment uses real physics data from The Well dataset.")

    gr.Markdown("## API Endpoints")
    gr.Markdown("- POST /reset - Reset environment with optional task parameter")
    gr.Markdown("- POST /step - Take action with action dict")
    gr.Markdown("- GET /state - Get current state")

demo.launch(server_port=7860)
