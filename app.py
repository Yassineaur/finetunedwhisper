from flask import Flask, request, jsonify
import torch
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperProcessor,
)
import os
from peft import PeftModel, PeftConfig


# Initialize the Flask app
app = Flask(__name__)

# Load model and processor
language = "arabic"
task = "transcribe"
base_model_name_or_path = "openai/whisper-small"

# Check if CUDA is available and set device to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

peft_config = PeftConfig.from_pretrained("YassineHamzaoui/whispernumbers")
model = WhisperForConditionalGeneration.from_pretrained(peft_config.base_model_name_or_path).to(device)
model = PeftModel.from_pretrained(model, "YassineHamzaoui/whispernumbers")

processor = WhisperProcessor.from_pretrained(peft_config.base_model_name_or_path)
tokenizer = WhisperTokenizer.from_pretrained(peft_config.base_model_name_or_path, language=language, task=task)
feature_extractor = processor.feature_extractor
forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
pipe = AutomaticSpeechRecognitionPipeline(model=model, tokenizer=tokenizer, feature_extractor=feature_extractor, device=0 if torch.cuda.is_available() else -1)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    audio_path = os.path.join('uploads', audio_file.filename)
    audio_file.save(audio_path)

    with torch.cuda.amp.autocast():
        text = pipe(audio_path, generate_kwargs={"forced_decoder_ids": forced_decoder_ids}, max_new_tokens=255)["text"]

    os.remove(audio_path)
    print(text)
    # Properly encode to JSON and then decode to handle Unicode
    return text

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)