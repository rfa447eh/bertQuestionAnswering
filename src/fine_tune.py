from transformers import BertForQuestionAnswering, BertTokenizer
import torch

def load_fine_tuned_model():
    # Load the saved model and tokenizer
    model = BertForQuestionAnswering.from_pretrained('./Fine_tuneBert')
    tokenizer = BertTokenizer.from_pretrained('./Fine_tuneBert')

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model, tokenizer, device

if __name__ == "__main__":
    model, tokenizer, device = load_fine_tuned_model()
    print("Fine-tuned model and tokenizer loaded successfully.")
