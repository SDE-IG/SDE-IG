## SDE-IG

### 1 Downloading weights
```angular2html
https://huggingface.co/hustcw/clap-asm/tree/main
```
put under dataset/clap_weights

### 2preprocessing
- 2.1 extract cfg and dfg

Please configure ghidra path before run this script
```angular2html
python extract_graph.py
```

- 2.2 extract fcg
```angular2html
python generate_fcg.py
```

- 2.3 generate pyg
```angular2html
python generate_ptg_data.py
```

### 3train
```angular2html
python main.py
```