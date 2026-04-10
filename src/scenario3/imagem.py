from graphviz import Digraph

dot = Digraph(format='png')
dot.attr(rankdir='TB', size='8,10')

# --- estilos ---
dot.attr('node', shape='box', style='filled', fontname="Helvetica")

# --- nós ---
dot.node('params', 'Parameter Sampling\n\nshake_threshold\nentropy_threshold\ncnn_weight\ntransformer_weight\nmin_steady_timesteps', fillcolor='#a6c8ff')  # azul

dot.node('sde', 'Streaming Decision Engine', fillcolor='#d0b3ff')  # roxo

dot.node('seq', 'Predicted Sequence\n(time series)', fillcolor='#e0e0e0')  # cinza

dot.node('steady', 'Steady Metrics\n\nAccuracy ↑\nPrecision ↑\nRecall ↑\nF1 ↑', fillcolor='#b6f2b6')  # verde

dot.node('transient', 'Transient Metrics\n\nTransition Precision ↑\nTransition Recall ↑\nMean Delay ↓\nMean Duration ↓\nFN Ratio ↓\nFP Ratio ↓', fillcolor='#ffd59e')  # laranja

dot.node('score', 'Score Function\n\nmaximize good\nminimize errors', fillcolor='#ffb3b3')  # vermelho

dot.node('opt', 'Optimization Loop\n(find best params)', fillcolor='#333333', fontcolor='white')  # preto

# --- ligações ---
dot.edge('params', 'sde')
dot.edge('sde', 'seq')

dot.edge('seq', 'steady')
dot.edge('seq', 'transient')

dot.edge('steady', 'score')
dot.edge('transient', 'score')

dot.edge('score', 'opt')
dot.edge('opt', 'params')  # loop

# --- gerar ---
dot.render('optimization_pipeline', view=True)  