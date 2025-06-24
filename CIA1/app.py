import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import os
import sacrebleu

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

st.title("Poem Generator using LLMs")

model_name = st.selectbox(
    "Choose a model:",
    ("llama3-70b-8192", "qwen/qwen3-32b", "mistral-saba-24b")
)

prompt = st.text_input("Enter your poem prompt...")
temperature = st.slider("Temperature (0.1 – 1.5):", 0.1, 1.5, 0.8)
top_p = st.slider("Top P (0.1 – 1.0):", 0.1, 1.0, 0.9)
max_tokens = st.slider("Max Length (50–2000 tokens):", 50, 2000, 700)
presence_penalty = st.slider("Presence Penalty (0.0 – 2.0):", 0.0, 2.0, 0.2)

reference_text = """The sky is painted with hues of gold,
As sunset's warmth begins to unfold.
The stars appear, like diamonds bright,
A night of rest, a peaceful sight.

The world is hushed, a quiet place,
Where worries fade, and love takes space.
The trees stand tall, their leaves rustling free,
A gentle breeze that whispers secrets to me.

The river flows, a winding stream,
That carries me to dreamland's theme.
The mountains rise, a majestic sight,
A challenge to climb, a beacon in flight.

In nature's arms, I find my peace,
A sense of calm, my worries release.
The world's vast beauty, it surrounds me still,
A symphony, that echoes through my will."""

def generate_poem(prompt, model, temperature, top_p, max_tokens):
    try:
        chat_completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a creative poem generator."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def calculate_bleu(reference_text, generated_text):
    bleu = sacrebleu.sentence_bleu(generated_text.strip(), [reference_text.strip()])
    return round(bleu.score, 2)

if st.button("Generate Poem"):
    if not prompt:
        st.warning("Please enter a prompt.")
    else:
        story = generate_poem(prompt, model_name, temperature, top_p, max_tokens)
        st.subheader("Generated Poem:")
        st.write(story)
        bleu = calculate_bleu(reference_text, story)
        st.markdown(f"**BLEU Score:** {bleu} (compared to reference poem)")

        st.subheader("Human Evaluation")
        coherence = st.slider("Coherence (1–5)", 1, 5, 3, key="coherence")
        creativity = st.slider("Creativity (1–5)", 1, 5, 4, key="creativity")
        fluency = st.slider("Fluency (1–5)", 1, 5, 3, key="fluency")

        st.markdown("Evaluation Summary")
        st.markdown(f"- **Coherence:** {coherence}/5")
        st.markdown(f"- **Creativity:** {creativity}/5")
        st.markdown(f"- **Fluency:** {fluency}/5")
        st.markdown(f"- **BLEU Score:** {bleu}")

