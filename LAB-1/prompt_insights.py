from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

prompts = [
    "Explain PCA.",
    "Explain PCA like I'm five.",
    "Explain PCA with a real-world analogy."
]

MODEL_NAME = "llama3-70b-8192"
def query_model(prompt, model=MODEL_NAME):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a highly descriptive assistant specialized in technical explanations."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=300,
        top_p=1,
    )
    return response.choices[0].message.content.strip()
def main():
    print(f"\nTesting Prompt Engineering with Model: {MODEL_NAME}\n")
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'='*80}")
        print(f"Prompt {i}: {prompt}")
        print(f"{'-'*80}")
        response = query_model(prompt)
        print(response)
        print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
