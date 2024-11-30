import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
import unicodedata
import re
import arabic_reshaper
from bidi.algorithm import get_display

# --- Load the AraGPT language model and tokenizer ---
model_name = 'aubmindlab/aragpt2-large'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

def normalize_text(text):
    """
    Normalize Arabic text using Unicode NFC normalization.
    """
    return unicodedata.normalize("NFC", text)

def prepare_rtl_text(text):
    """
    Reshape and reorder Arabic text for proper RTL display.
    """
    reshaped_text = arabic_reshaper.reshape(text)
    rtl_text = get_display(reshaped_text)
    return rtl_text

def calculate_surprisal_word_level(text, target_word):
    """
    Calculate surprisal score for a single target word in a given Arabic text.
    """
    # Normalize and reshape text
    text = normalize_text(text)
    target_word = normalize_text(target_word)

    # Display reshaped text for debugging
    rtl_text = prepare_rtl_text(text)
    print(f"RTL Text for Display: {rtl_text}")

    # Tokenize text
    tokenized_text = tokenizer.tokenize(text)
    encoded_text = tokenizer.encode(text, add_special_tokens=False)

    # Tokenize target word
    tokenized_target = tokenizer.tokenize(target_word)
    encoded_target = tokenizer.encode(target_word, add_special_tokens=False)

    print(f"Text: {text}")
    print(f"Target word: {target_word}")
    print(f"Tokenized Text: {tokenized_text}")
    print(f"Encoded Text: {encoded_text}")
    print(f"Tokenized Target: {tokenized_target}")
    print(f"Encoded Target: {encoded_target}")

    # Attempt to locate the target word in the encoded text
    try:
        target_start_index = -1
        for i in range(len(encoded_text) - len(encoded_target) + 1):
            if encoded_text[i:i + len(encoded_target)] == encoded_target:
                target_start_index = i
                break

        if target_start_index == -1:
            # Fallback to regex matching for token alignment
            regex = re.compile(r'\b' + re.escape(target_word) + r'\b')
            match = regex.search(text)
            if match:
                print(f"Fallback: Target word '{target_word}' found using regex.")
            else:
                raise ValueError(f"Target word '{target_word}' not found in the text.")
    except Exception as e:
        print(e)
        return None

    # Calculate surprisal for the first token of the target word
    with torch.no_grad():
        input_ids = torch.tensor([encoded_text]).to(torch.long)
        outputs = model(input_ids)
        logits = outputs.logits

    target_logits = logits[0, target_start_index]
    target_token_id = encoded_target[0]
    target_prob = torch.softmax(target_logits, dim=-1)[target_token_id].item()
    surprisal = -math.log2(target_prob)

    return surprisal

# --- Main Script ---
file_name = "full_stimuli.txt"

try:
    # Load the stimuli from the specified text file
    with open(file_name, "r", encoding="utf-8") as file:
        stimuli = [line.strip() for line in file if line.strip()]
except FileNotFoundError:
    print(f"Error: File '{file_name}' not found.")
    exit()

# Validate the file length
if len(stimuli) % 4 != 0:
    print("Error: The stimuli file must contain a multiple of 4 lines (4 sentences per stimulus).")
    exit()

# Initialize output
output_lines = []

# Process each stimulus
num_stimuli = len(stimuli) // 4
print(f"Total stimuli detected: {num_stimuli}")

for i in range(num_stimuli):
    stimulus_num = i + 1
    sentences = stimuli[i * 4: (i + 1) * 4]

    # Define conditions
    conditions = [
        "Plausible Sentence & Plausible Attractor",
        "Plausible Sentence & Implausible Attractor",
        "Implausible Sentence & Plausible Attractor",
        "Implausible Sentence & Implausible Attractor",
    ]

    output_lines.append(f"Stimulus {stimulus_num}:\n")
    print(f"Processing Stimulus {stimulus_num}...")

    for j, sentence in enumerate(sentences):
        # Display reshaped RTL sentence for debugging
        rtl_sentence = prepare_rtl_text(sentence)
        print(f"Processing Sentence (RTL Display): {rtl_sentence}")

        try:
            # Extract the target word dynamically using an Arabic comma
            target_word = normalize_text(sentence.split("،")[1].split()[0])
        except IndexError:
            print(f"Error: Unable to find target word in sentence: '{sentence}'")
            target_word = "N/A"

        surprisal_score = None
        if target_word != "N/A":
            surprisal_score = calculate_surprisal_word_level(sentence, target_word)

        # Format the output
        condition = conditions[j]
        if surprisal_score is not None:
            output_lines.append(f"  {condition}: Surprisal Score = {surprisal_score:.4f}\n")
        else:
            output_lines.append(f"  {condition}: Unable to calculate surprisal score (target word missing or invalid).\n")

    output_lines.append("\n")

# Write the output to a file
output_file_name = "Surprise_scores_word_level.txt"
with open(output_file_name, "w", encoding="utf-8") as output_file:
    output_file.writelines(output_lines)

print(f"Surprisal scores have been saved to '{output_file_name}'.")