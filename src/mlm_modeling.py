from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import pickle
from mlm_dataset import MLMDataset


def init_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.eos_token = tokenizer.sep_token
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    return tokenizer, model


def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return list(data)


def get_train_test_data(data, split=0.8):
    num_examples = len(data)
    train_data = data[:int(num_examples * split)]
    test_data = data[int(num_examples * split):]
    return train_data, test_data


def get_dataset(train_data, test_data, tokenizer):
    train_dataset = MLMDataset(train_data, tokenizer)
    test_dataset = MLMDataset(test_data, tokenizer)
    return {'train': train_dataset, 'test': test_dataset}

if __name__ == "__main__":
    print("Loading Model")
    tokenizer, model = init_model("bert-base-uncased")

    print("Loading Data")
    data = load_data("../kg_paths.pkl")

    print("Creating Dataset")
    train_data, test_data = get_train_test_data(data)
    lm_dataset = get_dataset(train_data, test_data, tokenizer)

    print(next(iter(lm_dataset["train"])))

    tokenizer.save_pretrained("custom_bert_mlm")

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15, mlm=True)

    training_args = TrainingArguments(
        output_dir="custom_bert_mlm",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        disable_tqdm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["test"],
        data_collator=data_collator,
    )

    trainer.train()
