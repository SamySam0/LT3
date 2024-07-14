''' This script generated synthetic medication prescriptions given medication names. '''
import torch, json, argparse
from transformers import BertTokenizer
from datasets import Dataset
from model.As_BeamTranslator import Translator
from model.Models import Transformer
from modules.Generator import Generator
import json


if __name__ == "__main__":

    # ----- FETCH COMMANDLINE ARGUMENTS -----
    parser = argparse.ArgumentParser(description="Select options for generationg synthetic data.")
    parser.add_argument("-in",  "--input_path", type=str, required=False, default="medications.txt", help="Path to input dictionnary's file.")
    parser.add_argument("-out", "--output_path", type=str, required=False, default="generations.json", help="Path to output file.")

    parser.add_argument("-bs",   "--beam_size", type=int, required=False, default=4, help="Beam size for beam search decoding.")
    parser.add_argument("-mspd", "--max_step_prob_diff", type=float, required=False, default=1.0, help="Maximal step probability difference for beam search decoding.")
    parser.add_argument("-nrpl", "--no_repetition_length", type=int, required=False, default=4, help="Minimal length between repeated special characters in generations.")
    parser.add_argument("-a",    "--alpha", type=float, required=False, default=0.6, help="Alpha value (hyperparameter) for beam search decoding.")
    parser.add_argument("-tlp",  "--tree_length_product", type=int, required=False, default=3, help="Tree length product for beam search decoding.")
    args = parser.parse_args()

    # ----- FETCH MEDICATION DATA -----

    # Fetch medications to generate
    medication_names = {}

    INPUT_FILE = args.input_path
    with open(INPUT_FILE, 'r') as file:
        for line in file:
            med, amount = line.strip().split(':')
            medication_names[med] = int(amount)

    dataset = Dataset.from_dict({"medication": medication_names.keys(), "amount": medication_names.values()})


    # ----- BUILD DATASET -----

    # Load pre-trained tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    # Tokenize medication dataset to generate
    def preprocessing(samples, tokenizer, max_input_length):
        samples["input_ids"] = tokenizer(samples["medication"], max_length=max_input_length+1, truncation=True, padding="max_length")["input_ids"] + [tokenizer.pad_token_id]
        samples["input_ids"].remove(tokenizer.sep_token_id)
        samples["input_ids"].remove(tokenizer.cls_token_id)
        return samples

    tokenized_dataset = dataset.map(lambda x: preprocessing(x, tokenizer, max_input_length=24))


    # ----- LOAD LT3 MODEL -----

    # Verify whether GPU is available to host model inference
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load LT3 model architecture
    transformer = Transformer(
        keyword_max_length = 24, description_max_length = 155,
        vocab_size = 28996, pad_idx = 0,
        d_model = 515, d_v = 64, d_hid = 2038, n_head = 5, n_layers = 2,
        dropout = 0.2,
    )

    # Load pretrained LT3 weights
    MODEL_PATH = "lt3_model.pt"
    transformer.load_state_dict(torch.load(MODEL_PATH))

    # Fetch decoding parameters (from user argparse)
    BEAM_SIZE = args.beam_size
    MAXIMAL_STEP_PROB_DIF = args.max_step_prob_diff
    NRP_LEN = args.no_repetition_length
    ALPHA = args.alpha
    TREE_LEN_PRODUCT = args.tree_length_product

    # Initalised LT3 decoder (B2SD, etc.)
    model = Translator(
        transformer=transformer, 
        pad_idx=0, sos_idx=101, eos_idx=102, 
        max_output_length=155, 
        beam_size=BEAM_SIZE, maximal_step_probability_difference=MAXIMAL_STEP_PROB_DIF,
        nrp_length=NRP_LEN, alpha=ALPHA, tree_length_product=TREE_LEN_PRODUCT,
    ).to(device)


    # ----- GENERATE SYNTHETIC DATA -----

    # Generate prescriptions for input medications
    predictions = Generator(tokenized_dataset, tokenizer).generate(translator=model, device=device)

    # Write generated data to file
    OUTPUT_FILE = args.output_path
    assert OUTPUT_FILE.endswith('.json'), "Output file must be a .json!"

    with open(OUTPUT_FILE, 'w') as f:
        f.write(json.dumps(predictions, indent=2, separators=(',', ': ')).replace('],', '],\n'))

    print(f'Predictions successfully written to {OUTPUT_FILE} .')
