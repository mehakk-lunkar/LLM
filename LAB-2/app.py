import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import os
import sacrebleu

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

st.title("Story Generator using LLMs")

model_name = st.selectbox(
    "Choose a model:",
    ("llama3-70b-8192", "qwen/qwen3-32b", "mistral-saba-24b")
)

prompt = st.text_input("Enter your story prompt...")
temperature = st.slider("Temperature (0.1 – 1.5):", 0.1, 1.5, 0.8)
top_p = st.slider("Top P (0.1 – 1.0):", 0.1, 1.0, 0.95)
max_tokens = st.slider("Max Length (50–1500 tokens):", 50, 1500, 700)
presence_penalty = st.slider("Presence Penalty (0.0 – 2.0):", 0.0, 2.0, 0.0)

reference_text = """Elira held the letter with trembling hands. The parchment was soft and yellowed with age, ...
... “To the last of the Starbound Line…”

The truth, long-buried, was beginning to rise. And Elderglen, sleepy and silent, would soon awaken to a war older than time."""

def generate_story(prompt, model, temperature, top_p, max_tokens):
    try:
        chat_completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a creative storyteller."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def calculate_bleu(reference_text, generated_text):
    bleu = sacrebleu.sentence_bleu(generated_text.strip(), [reference_text.strip()])
    return round(bleu.score, 2)

if st.button("Generate Story"):
    if not prompt:
        st.warning("Please enter a prompt.")
    else:
        story = generate_story(prompt, model_name, temperature, top_p, max_tokens)
        st.subheader("Generated Story:")
        st.write(story)

        # Evaluation
        bleu = calculate_bleu(reference_text, story)
        st.markdown(f"**BLEU Score:** {bleu} (compared to reference story)")

        st.subheader("Human Evaluation")
        coherence = st.slider("Coherence (1–5)", 1, 5, 3, key="coherence")
        creativity = st.slider("Creativity (1–5)", 1, 5, 4, key="creativity")
        fluency = st.slider("Fluency (1–5)", 1, 5, 3, key="fluency")

        st.markdown("Evaluation Summary")
        st.markdown(f"- **Coherence:** {coherence}/5")
        st.markdown(f"- **Creativity:** {creativity}/5")
        st.markdown(f"- **Fluency:** {fluency}/5")
        st.markdown(f"- **BLEU Score:** {bleu}")

