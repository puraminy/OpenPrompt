
# # Conditional Generation with Prefix Tuning.
# In this tutorial, we do conditional generation with prefix tuning template.

# we use WebNLG as an example, as well. Note that the evaluation of generation result should be done
# by using the scripts provided by https://github.com/Yale-LILY/dart/tree/master/evaluation, 
# Which we do not include in it. 

import argparse
import torch
from openprompt.utils.logging import logger
from comet.train.eval import *
from comet.train.common import *
from openprompt.data_utils.atomic import ATOMICProcessor

import click

@click.command()
@click.option(
    "--lr",
    "-",
    default=5e5,
    type=float,
    help=""
)
@click.option(
    "--plm_eval_mode",
    "-",
    is_flag=True,
    help=""
)
@click.option(
    "--model",
    default="t5",
    type=str,
    help=""
)
@click.option(
    "--model_name_or_path",
    "-mp",
    default="t5_base",
    type=str,
    help=""
)
@click.option(
    "--extend_tok",
    "-et",
    is_flag=True,
    help=""
)
@click.option(
    "--num_samples",
    "-n",
    default=100,
    type=int,
    help=""
)
@click.option(
    "--val_samples",
    "-ng",
    default=50,
    type=int,
    help=""
)
def main(lr, plm_eval_mode, model_name_or_path, extend_tok, model, train_samples, val_samples):
    dataset = {}
    ap = ATOMICProcessor()
    dataset['train'] = ap.get_train_examples("../experiments/db_atomic/", train_samples)
    dataset['validation'] = ap.get_dev_examples("../experiments/db_atomic/", val_samples)
    dataset['test'] = ap.get_test_examples("../experiments/db_atomic/", val_samples)

    dataset['train'] = dataset['train'][:train_samples]
    dataset['validation'] = dataset['validation'][:val_samples]
    dataset['test'] = dataset['test'][:val_samples]

    # load a pretrained model, its tokenizer, its config, and its TokenzerWrapper by one function 
    from openprompt.plms import load_plm
    plm, tokenizer, model_config, WrapperClass = load_plm(model, model_name_or_path)
    if extend_tok:
        tokenizer = extend_tokenizer(tokenizer, "xIntent")

    # Instantiating the PrefixTuning Template !
    from openprompt.prompts.prefix_tuning_template import PrefixTuningTemplate
    # we can use a plain text as the default setting
    # i.e. 
    # mytemplate = PrefixTuningTemplate(model=plm, tokenizer=tokenizer)
    # is equal to 
    # mytemplate = PrefixTuningTemplate(model=plm, tokenizer=tokenizer, text='{"placeholder":"text_a"} {"mask"}')
    #mytemplate = PrefixTuningTemplate(model=plm,  tokenizer=tokenizer, text='{"soft"} {"soft"} {"soft"} {"soft"} {"placeholder":"text_a"} {"special": "<eos>"} {"mask"} ', using_decoder_past_key_values=False)

    mytemplate = PrefixTuningTemplate(model=plm,  tokenizer=tokenizer, text='{"placeholder":"text_a"} {"special": "<eos>"} {"mask"} ', using_decoder_past_key_values=False)
    # To better understand how does the template wrap the example, we visualize one instance.
    # You may observe that the example doesn't end with <|endoftext|> token. Don't worry, adding specific end-of-text token
    # is a language-model-specific token. we will add it for you in the TokenizerWrapper once you pass `predict_eos_token=True`
    an_example = dataset['train'][0]
    print(an_example)

    wrapped_example = mytemplate.wrap_one_example(an_example) 
    print(wrapped_example)


    # Your can loop over the dataset by yourself by subsequently call mytemplate.wrap_one_example  and WrapperClass().tokenizer()
    # but we have provide a PromptDataLoader for you.
    from openprompt import PromptDataLoader
    train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer, 
        tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=256, 
        batch_size=5,shuffle=True, teacher_forcing=True, predict_eos_token=True, # be sure to pass predict_eos_token=True if your tempalte doesn't contain one, or you model may fail to stop generation.
        truncate_method="head")

    validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer, 
        tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=256, 
        batch_size=5,shuffle=False, teacher_forcing=False, predict_eos_token=True,
        truncate_method="head")

    test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer, 
        tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=256, 
        batch_size=5,shuffle=False, teacher_forcing=False, predict_eos_token=True,
        truncate_method="head")

    # load the pipeline model PromptForGeneration.
    from openprompt import PromptForGeneration
    use_cuda = True
    prompt_model = PromptForGeneration(plm=plm,template=mytemplate, freeze_plm=True,tokenizer=tokenizer, plm_eval_mode=plm_eval_mode)
    if use_cuda:
        prompt_model=  prompt_model.cuda()


    from transformers import AdamW
    # Follow PrefixTuning（https://github.com/XiangLi1999/PrefixTuning), we also fix the language model
    # only include the template's parameters in training. 

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
    {
        "params": [p for n, p in mytemplate.named_parameters() if (not any(nd in n for nd in no_decay)) and p.requires_grad],
        "weight_decay": 0.0,
    },
    {
        "params": [p for n, p in mytemplate.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
        "weight_decay": 0.0,
    },
    ]


    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)

    from transformers.optimization import get_linear_schedule_with_warmup

    tot_step  = len(train_dataloader)*5
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, tot_step)

    # We provide generation a generation metric, you can also define your own. Note that it's not directly comparable to WebNLG's scripts evaluation.
    from openprompt.utils.metrics import generation_metric
    # Define evaluate function 
    def evaluate(prompt_model, dataloader):
        generated_sentence = []
        groundtruth_sentence = []
        prompt_model.eval()

        for step, inputs in enumerate(dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            _, output_sentence = prompt_model.generate(inputs, **generation_arguments)
            generated_sentence.extend(output_sentence)
            groundtruth_sentence.extend(inputs['tgt_text'])
        score = generation_metric(generated_sentence, groundtruth_sentence, "sentence_bleu")
        print("test_score", score, flush=True)
        return generated_sentence

    generation_arguments = {
        "max_length": 512,
        "max_new_tokens": None,
        "min_length": 5,
        "temperature": 1.0,
        "do_sample": False,
        "top_k": 0,
        "top_p": 0.9,
        "repetition_penalty": 1.0,
        "num_beams": 5,
        "bad_words_ids": [[628], [198]]
    }

    # training and generation.
    global_step = 0 
    tot_loss = 0 
    log_loss = 0
    for epoch in range(1):
        prompt_model.train()
        for step, inputs in enumerate(train_dataloader):
            global_step +=1
            if use_cuda:
                inputs = inputs.cuda()
            loss = prompt_model(inputs)
            loss.backward()
            tot_loss += loss.item()
            mean_loss = tot_loss / global_step
            print("Epoch {}, global_step {} average loss: {:.2f} ".format(epoch, global_step, mean_loss), flush=True)
            torch.nn.utils.clip_grad_norm_(mytemplate.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if global_step %500 ==0: 
                print("Epoch {}, global_step {} average loss: {} lr: {}".format(epoch, global_step, (tot_loss-log_loss)/500, scheduler.get_last_lr()[0]), flush=True)
                log_loss = tot_loss

    generated_sentence = evaluate(prompt_model, test_dataloader)
    val_data = ap.get_val_data()
    inter = False
    save_path = ""
    output_name = "op_results"
    gen_param = "greedy"
    val_records=len(dataset["validation"])
    #prompt_model.eval()
    #eval(prompt_model, tokenizer, val_data, inter, save_path, output_name, val_records, gen_param)  

    #with open(f"results_{plm_eval_mode}.txt",'w') as f:
    #    for i in generated_sentence:
    #        f.write(i+"\n")
if __name__ == "__main__":
   main()
