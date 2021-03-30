from seq2seq_model import Seq2SeqModel, Seq2SeqArgs

import pandas as pd


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


model = Seq2SeqModel(encoder_type="bert", encoder_name="outputs/best_model/encoder",
                     decoder_name="outputs/best_model/encoder",
                     args=model_args, use_cuda=False)

test = pd.read_csv("test.csv")

def add_tag(text):
  return "[SEP] " + text + " [SEP]"

test['input_text'] = test['raw_address'].apply(add_tag)

test['predicted'] = model.predict(test['input_text'])

def process_final_result(pred):
    if (pred.strip()).endswith("/"):
      a = pred.split('/')[0].strip() + "/"
      return a
    else:
      if len(pred.split('/')) > 1:
        a = pred.split('/')[0].strip() + '/'+ pred.split('/')[1].strip()
        return a
      elif len(pred.split('/')) == 1:
        a = '/'+ pred.split('/')[0].strip()
        return a
      else:
        a = pred.strip()
        return a

test['POI/street'] = test['predicted'].apply(process_final_result)

submissions = pd.DataFrame({"id": test['id'], "POI/street": test['POI/street']})
submissions.to_csv("Submission.csv",index=False)



