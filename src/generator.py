from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")

def generate(query, context):
    prompt = f"Context: {context}\nQuestion: {query}"

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return res.choices[0].message.content
