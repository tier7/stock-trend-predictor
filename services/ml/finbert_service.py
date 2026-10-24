from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "ProsusAI/finbert"
MODEL_REVISION = "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031"

def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model.eval()
    return tokenizer, model

def analyze_sentiment(text, tokenizer, model):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text must be a non-empty string")

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1)[0]

    result = {}
    for index,prob in enumerate(probabilities):
        class_name = model.config.id2label[index]
        result[class_name] = prob.item()
    return result

class FinbertAnalyzer:
    def __init__(self):
        self.tokenizer = None
        self.model = None

    def analyze(self, text):
        if self.model is None:
            self.tokenizer, self.model = load_finbert()

        return analyze_sentiment(
            text, self.tokenizer, self.model)

if __name__ == "__main__":
    tokenizer, model = load_finbert()
    texts = [
        "The company reported record profits and raised its annual forecast.",
        "The company reported a large loss and cut its annual forecast.",
        "The company will publish its quarterly results on Tuesday.",
    ]
    for text in texts:
        sentiment = analyze_sentiment(text, tokenizer, model)
        print(text, sentiment)
