from transformers import MarianMTModel, MarianTokenizer

def translate_to_english(text, source_lang="id", target_lang="en"):
    # set up marianMT model for translating 
    model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
