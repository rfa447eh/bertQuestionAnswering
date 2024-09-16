import pytest
from transformers import BertForQuestionAnswering, BertTokenizer
import torch

# Load the saved model and tokenizer
@pytest.fixture(scope="module")
def qa_model():
    model = BertForQuestionAnswering.from_pretrained('/content/bert')
    tokenizer = BertTokenizer.from_pretrained('/content/bert')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set model to evaluation mode
    return model, tokenizer, device

def get_answer(question, context, model, tokenizer, device):
    max_length = 512
    context_chunks = [context[i:i + max_length] for i in range(0, len(context), max_length)]

    best_answer = ""
    best_score = float('-inf')
    for chunk in context_chunks:
        encoding = tokenizer.encode_plus(
            question, chunk, 
            add_special_tokens=True, 
            max_length=max_length, 
            truncation=True, 
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(device)
        token_type_ids = encoding['token_type_ids'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, token_type_ids=token_type_ids)

        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        start_index = torch.argmax(start_scores)
        end_index = torch.argmax(end_scores)

        answer_tokens = tokenizer.convert_ids_to_tokens(input_ids[0][start_index:end_index + 1])
        answer = tokenizer.convert_tokens_to_string(answer_tokens)

        answer_score = start_scores[0][start_index] + end_scores[0][end_index]
        if answer_score > best_score:
            best_score = answer_score
            best_answer = answer

    return best_answer

# Test if model and tokenizer load correctly
def test_model_loading(qa_model):
    model, tokenizer, device = qa_model
    assert model is not None, "Model failed to load."
    assert tokenizer is not None, "Tokenizer failed to load."
    assert device is not None, "Device not found."

# Test the get_answer function
def test_get_answer_function(qa_model):
    model, tokenizer, device = qa_model
    
    question = "What does the Matters feature in RunSensible allow you to do when managing legal cases?"
    context = "RunSensible’s Matters feature offers a centralized system for managing all aspects of legal cases, allowing you to organize and track important details in one place."

    answer = get_answer(question, context, model, tokenizer, device)
    
    expected_answer = "organize and track important details in one place"
    assert answer == expected_answer, f"Expected: {expected_answer}, but got: {answer}"

# Test when context is too long (chunking)
def test_long_context(qa_model):
    model, tokenizer, device = qa_model
    
    question = "How do you create a new matter in the system?"
    context = "To create a new matter in the system, begin by clicking the New button located in the top right corner of the screen." * 100  # Simulate a long context

    answer = get_answer(question, context, model, tokenizer, device)
    
    # We don't know the exact answer in this case but we can at least test that we don't get an empty answer
    assert answer is not None and len(answer) > 0, "Answer is empty for a long context."

