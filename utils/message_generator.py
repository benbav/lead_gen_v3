# utils/message_generator.py
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_message(client_info):
    # Load pre-trained LLaMA-2 model and tokenizer
    model_name = "meta-llama/Llama-2-7b"  # Replace with the correct model name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Few-shot learning: Provide therapy-specific examples to guide the model
    examples = """
    Example 1:
    Input: Write a personalized outreach message for a therapy business owner named Sarah.
    Output: Hi Sarah, I came across your therapy practice and love how you focus on mindfulness-based techniques. I specialize in creating dashboards that help businesses like yours track client progress and streamline operations. Would you be open to a quick chat about how I can help?

    Example 2:
    Input: Write a personalized outreach message for a therapy business owner named Michael.
    Output: Hi Michael, I noticed your work with couples therapy and how you emphasize communication skills. I help therapy practices like yours visualize key metrics and improve client outcomes through custom dashboards. Let me know if you'd like to explore how this could work for your practice!

    Example 3:
    Input: Write a personalized outreach message for a therapy business owner named Emily.
    Output: Hi Emily, I saw your therapy practice specializes in trauma-informed care. I work with therapists to create dashboards that simplify tracking client progress and session outcomes. Would you be interested in learning more about how this could save you time and enhance your practice?
    """

    # Craft the prompt with few-shot examples and the new input
    prompt = f"""
    {examples}

    Input: Write a personalized outreach message for a therapy business owner named {client_info['name']}.
    Output:
    """

    # Tokenize the prompt and generate the message
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=200)
    message = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the generated message (remove the prompt)
    message = message.split("Output:")[-1].strip()
    return message
