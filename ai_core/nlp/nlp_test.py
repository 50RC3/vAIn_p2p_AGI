from ai_core.nlp.nltk_utils import analyze_text_nltk, initialize_nltk_components

initialize_nltk_components()
text = "This is a sample text for testing NLP processing. GPT models are transforming AI capabilities worldwide."
result = analyze_text_nltk(text)
print("NLP Analysis Result:", result)