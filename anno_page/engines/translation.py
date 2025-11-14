from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from anno_page.engines import BaseEngine


class TranslationEngine(BaseEngine):
    def __init__(self, config, device, config_path):
        super().__init__(config, device, config_path)

        self.tokenizer_name = self.config["TOKENIZER"]
        self.model_name = self.config["MODEL"]

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)

    def process(self, texts: str | list[str]) -> list[str]:
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model.generate(**inputs)
        translated_texts = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        return translated_texts
