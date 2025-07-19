# Part-Of-Speech Prediction Evaluation

Example command to evaluate a pretrained model like GPT2 taken from huggingface is:
```
python eval_pos_pred.py --pretrained "gpt2"
```

Running the following commands will get the result.
```
>>> from llm_devo.notebooks.pos_pred_metrics import PosPredResults
>>> test = PosPredResults(model_name='pretrained/gpt2')
>>> test.get_aggre_perf()
(0.9026763990267641, None)
```

Masked models can be tested in the same way.
