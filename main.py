import spacy
import tensorflow as tf
from transformers import TFT5ForConditionalGeneration, RobertaTokenizer

# Load the spaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Load the saved finetuned model
model = TFT5ForConditionalGeneration.from_pretrained(args.save_dir)
# Load the saved tokenizer
tokenizer = RobertaTokenizer.from_pretrained(args.save_dir)

def generate_python_code(args, prompt_text, max_tokens=150):
    try:
        # Encode the input text by prepending the task for input sequence
        encoded_prompt = args.prefix + prompt_text
        encoded_prompt = tokenizer(encoded_prompt, return_tensors="tf", padding="max_length", truncation=True, max_length=args.max_input_length)

        # Inference - Generate code using the model
        generated_code = model.generate(
            encoded_prompt["input_ids"], attention_mask=encoded_prompt["attention_mask"],
            max_length=max_tokens, top_p=0.7, top_k=50, repetition_penalty=2.0, num_return_sequences=1
        )

        # Decode generated tokens to get the final code
        decoded_code = tokenizer.decode(generated_code.numpy()[0], skip_special_tokens=True)
        return decoded_code

    except Exception as e:
        # Handle any errors 
        print("Error:", e)
        return None

def contains_python(prompt_text):
    # Use spaCy to check if the input contains "python"
    doc = nlp(prompt_text.lower())
    return any(token.text == "python" for token in doc)

# Example prompt
prompt_text = input("Enter the code to print:\n")

if contains_python(prompt_text):
    # Generate Python code using the prompt
    generated_code = generate_python_code(args, prompt_text)

    if generated_code is not None:
        # Print the generated code if it's not None
        print("Generated code:\n", generated_code)
    else:
        print("Sorry, I don't know.")
else:
    print("Sorry, I don't know.")


#Proudly created by Atharva Patel
#Proudly created by Aylin Ambriz
