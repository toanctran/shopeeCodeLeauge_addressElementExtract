from seq2seq_model import Seq2SeqArgs, Seq2SeqDataset, Seq2SeqModel

import pandas as pd

df = pd.read_csv("train.csv")


def add_tag(text):
  return "[SEP] " + text + " [SEP]"



df[['POI', 'street']] = df['POI/street'].str.split('/', 1, expand=True)
df['POI'].replace('', '/', inplace=True)
df['street'].replace('', '/', inplace=True)
df['raw_address'] = df['raw_address'].apply(add_tag)
df['POI/street'] = df['POI/street'].apply(add_tag)

X = df['raw_address']
y = df['POI/street']

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.04, random_state=42)
train_ds = pd.DataFrame({'input_text': X, 'target_text':y})
val_ds = pd.DataFrame({'input_text': X_val, 'target_text':y_val})


model_args = Seq2SeqArgs()
model_args.num_train_epochs =10
model_args.use_multiprocessing = False
model_args.use_multiprocessed_decoding = False
model_args.evaluate_generated_text = True
model_args.evaluate_during_training = True
model_args.evaluate_during_training_verbose = True
model_args.evaluate_during_training_silent = False
model_args.evaluate_during_training_steps = 0
model_args.overwrite_output_dir = True
model_args.early_stopping_consider_epochs = True
model_args.use_early_stopping = True
model_args.use_cached_eval_features = True
model_args.train_batch_size = 32
model_args.save_steps = 0
model_args.early_stopping_metric = "matches"
model_args.early_stopping_metric_minimize = False
model_args.output_dir = "outputs_v2/"
model_args.weight_decay = 0.01
model_args.learning_rate = 3e-5


model = Seq2SeqModel(encoder_type="bert", encoder_name="cahya/bert-base-indonesian-522M",
                     decoder_name="cahya/bert-base-indonesian-522M",
                     args=model_args, use_cuda=False)

def count_matches(labels, preds):
    # print(labels)
    # print(preds)
    predictions = []
    for pred in preds:
      if (pred.strip()).endswith("/"):
        a = "[SEP] " + pred.split('/')[0].strip() + "/ [SEP]"
        predictions.append(a)
      else:
        if len(pred.split('/')) > 1:
          a = "[SEP] " + pred.split('/')[0].strip() + '/'+ pred.split('/')[1].strip() + " [SEP]"
          predictions.append(a)
        elif len(pred.split('/')) == 1:
          a = "[SEP] " + '/'+ pred.split('/')[0].strip() + " [SEP]"
          predictions.append(a)
        else:
          a = "[SEP] " + pred.strip() + " [SEP]"
          predictions.append(a)

      
    result = [
            1 if label == pred else 0
            for label, pred in zip(labels, predictions)
        ]
    
    print(f'Matches: {sum(result)} (or {(sum(result)*100/len(result)):.2f}% of validation dataset)')
    return sum(result)




model.train_model(train_data=train_ds,verbose=True, output_dir='outputs', matches=count_matches, eval_data=val_ds)