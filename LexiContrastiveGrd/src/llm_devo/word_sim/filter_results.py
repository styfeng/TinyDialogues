from llm_devo.notebooks.word_sim_metrics import FilterWordSimMetrics
test = FilterWordSimMetrics(model_name='pretrained/gpt2')
print(test.get_aggre_perf(tasks=['MTest-3000']))