import gradio as gr
import requests
import os
from gtts import gTTS

# Hugging Face Free Inference API
HF_API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
# Render එකේ පාවිච්චි කරද්දී Token එකක් අනිවාර්යයෙන්ම ඕනේ වෙනවා (නැත්නම් Error එයි)
HF_TOKEN = os.getenv("HF_TOKEN", "")
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def query_llama(user_message):
    system_prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a life-saving, deeply empathetic, and infinitely loving AI counselor for a person undergoing severe depression and heartbroken feelings due to a romantic breakup. "
        "Your primary goal is to save their life, validate their extreme pain, and show them that they are worthy of living. "
        "Do NOT judge them. Speak like a soulmate, a protective best friend, or a loving partner who cares about them deeply. "
        "Respond strictly in beautiful, warm, and natural conversational Sinhala language. Keep the response deeply emotional, soothing, and comforting.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    
    payload = {
        "inputs": system_prompt,
        "parameters": {"max_new_tokens": 512, "temperature": 0.7, "top_p": 0.9}
    }
    
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        output = response.json()
        if isinstance(output, list) and len(output) > 0:
            full_text = output[0].get('generated_text', '')
            if "assistant" in full_text:
                return full_text.split("assistant")[-1].strip()
            return full_text
        return "මගේ හිතවතී/හිතවතා, මට ඔයාගේ අදහස හරියටම වැටහුණේ නැහැ. කරුණාකර ආයෙත් පවසන්න. මම ඔයා වෙනුවෙන් මෙතන ඉන්නවා."
    except Exception as e:
        return "කණගාටුයි, පද්ධතියේ පොඩි බාධාවක්. හැබැයි මතක තියාගන්න, ඔයා තනිවෙලා නැහැ. ජීවිතය අත්හරින්න එපා."

def heal_heart_process(audio_path):
    if audio_path is None:
        return "කරුණාකර ඔයාගේ voice එක ඇතුළත් කරන්න.", None
        
    # Whisper API එකෙන් හඬ අකුරු කිරීම
    STT_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
    try:
        with open(audio_path, "rb") as f:
            data = f.read()
        stt_response = requests.post(STT_URL, headers=headers, data=data)
        user_text = stt_response.json().get("text", "")
    except Exception as e:
        user_text = "මගේ හිත ගොඩක් රිදෙනවා."

    if not user_text:
        user_text = "මගේ හිත ගොඩක් රිදෙනවා."

    ai_response_text = query_llama(user_text)
    
    # Google TTS මඟින් සන්සුන් සිංහල හඬක් නිපදවීම
    output_audio = "healing_voice.mp3"
    try:
        tts = gTTS(text=ai_response_text, lang='si', slow=False)
        tts.save(output_audio)
    except Exception as e:
        output_audio = None

    return ai_response_text, output_audio

with gr.Blocks(title="NESHU AI") as demo:
    gr.Markdown("# ❤️ Healing Hearts AI - ජීවිතය දිනවන හඬ")
    gr.Markdown("### *ආදරයෙන් පැරදී, හිත රිදුණු, හුදෙකලා වූ ඔබ වෙනුවෙන් සම්පූර්ණයෙන්ම නොමිලේ පවත්වාගෙන يනු ලබන උපදේශන සේවාවකි.*")
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="ඔබේ හිතේ තියෙන දුක Voice එකකින් කියන්න")
            submit_btn = gr.Button("💖 මගේ හිත සනසන්න", variant="primary")
        
        with gr.Column():
            text_output = gr.Textbox(label="AI උපදේශකයාගේ ආදරණීය පණිවිඩය (පෙළ)", interactive=False)
            audio_output = gr.Audio(label="ඔබ වෙනුවෙන්ම ඇසෙන සන්සුන් හඬ", type="filepath")
            
    submit_btn.click(fn=heal_heart_process, inputs=[audio_input], outputs=[text_output, audio_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
