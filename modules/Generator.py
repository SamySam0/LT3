import torch, re

class Generator:

    def __init__(self, tokenized_dataset, tokenizer):
        self.tokenized_dataset = tokenized_dataset
        self.tokenizer = tokenizer
  
    def postprocessing(self, sentence):
        # Specific cases
        sentence = sentence.replace("[** ", "[**").replace(" **]", "**]")
        sentence = sentence.replace(" : * ", ":*").replace(" : ", ": ").replace(' *', '*')

        # Stick to left (end of word)
        for p in "!),.;?]}":
            sentence = sentence.replace(f" {p}", f"{p}")

        # Stick to right (start of word)
        for p in "#([{":
            sentence = sentence.replace(f"{p} ", f"{p}")

        # Stick to both
        for p in "'\/-":
            sentence = sentence.replace(f" {p} ", f"{p}")

        # Stick to none (leave space)
        for p in "%&\"<=>":
            pass

        # Remove (or useless)
        for p in "":
            sentence = sentence.replace(f"{p}", "")

        # Specific cases
        sentence = sentence.replace(" (s)", "(s)").replace("(E. C.)", "(E.C.)")
        sentence = re.sub(r'\b(\d+)\s*\.\s*(\d+)\b', r'\1.\2', sentence)
        sentence = re.sub(r'\b(\d+)\s*\,\s*(\d+)\b', r'\1.\2', sentence)

        return ' '.join(sentence.split())
  
    def predict(self, translator, src_tokens, device, nb_outputs):
        predictions = translator.translate_sentence(src_tokens.unsqueeze(0).to(device), nb_outputs=nb_outputs)
        untokenized_predictions = self.tokenizer.batch_decode(predictions)
        untokenized_predictions = [self.postprocessing(pred) for pred in untokenized_predictions]
        return untokenized_predictions
  
    def generate(self, translator, device):
        predictions = {}
        for sample in self.tokenized_dataset:
            untokenized_predictions = self.predict(translator, src_tokens=torch.tensor(sample['input_ids']), device=device, nb_outputs=sample['amount'])

            pred_len, required_len = len(untokenized_predictions), sample['amount']
            if pred_len < required_len:
                untokenized_predictions.extend(["-"] * (required_len - pred_len))
            
            assert sample['medication'] not in predictions, \
                "Medication already generated. Make sure you didn't enter the same medication twice."
            predictions[sample['medication']] = untokenized_predictions
            
        return predictions