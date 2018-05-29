### Available models: 

- **Graves2006:** see the reference at:[Graves' model](ftp://ftp.idsia.ch/pub/juergen/icml2006.pdf)

- **eyben:** see the reference at:[Eybens' model](http://ieeexplore.ieee.org/abstract/document/5373257/)

- **maas:** see the reference at:[Maas' model](http://www.aclweb.org/anthology/N15-1038)


- **deep_speech:** see the reference at:[Deep Speech model](https://arxiv.org/abs/1412.5567)


- **deep_speech2:** see the reference at:[Deep Speech2 model](https://arxiv.org/abs/1512.02595)

- **brsmv1:** BRSM v1.0 model


## Example of training for all available models:



### BRSM v1.0 model (default model)

This model is trained using MFCC, using 13 coefficients, also  use Delta and Delta-Delta.

You must preprocess the dataset in an hdf5 file by using the following parameters :

For Brazilian Portuguese Speech Dataset(BRSD) Click [here](datasets.md) for more information:
```
python -m extras.make_dataset --parser brsd  --input_parser mfcc --override

```

For English Speech Dataset (ENSD) Click [here](docs/datasets.md) for more information.
```
python -m extras.make_dataset --parser ensd  --input_parser mfcc  --override

```

Train the model using this command:

For Brazilian Portuguese Speech Dataset(BRSD):

```   
python train.py --dataset .datasets/brsd/data.h5 

```
For English Speech Dataset (ENSD):


```   
python train.py --dataset .datasets/ensd/data.h5 

```


### Graves model

This model is trained using MFCC, using 26 coefficients, also does not use Delta and Delta-Delta.

You must preprocess the dataset in an hdf5 file by using the following parameters :

For Brazilian Portuguese Speech Dataset(BRSD) Click [here](datasets.md) for more information:
```
python -m extras.make_dataset --parser brsd  --input_parser mfcc --input_parser_params "num_cep 26 dd 0 d 0" --override

```

For English Speech Dataset (ENSD) Click [here](docs/datasets.md) for more information.
```
python -m extras.make_dataset --parser ensd  --input_parser mfcc --input_parser_params "num_cep 26 dd 0 d 0" --override

```


Train the model using this command:

For Brazilian Portuguese Speech Dataset(BRSD):

```   
python train.py --dataset .datasets/brsd/data.h5 --model graves2006

```
For English Speech Dataset (ENSD):


```   
python train.py --dataset .datasets/ensd/data.h5 --model graves2006

```



### Deep Speech2

This model is trained using  Mel Spectrogram (LogFilterBank), using 161 filts  calculated on 20ms windows.

You must preprocess the dataset in an hdf5 file by using the following parameters :

For Brazilian Portuguese Speech Dataset(BRSD) Click [here](datasets.md) for more information:
```
python -m extras.make_dataset --parser brsd  --input_parser logfbank --input_parser_params "win_len 0.02 num_filt 161" --override

```

For English Speech Dataset (ENSD) Click [here](docs/datasets.md) for more information.
```
python -m extras.make_dataset --parser ensd  --input_parser logfbank --input_parser_params "win_len 0.02 num_filt 161" --override


```

Train the model using this command:

For Brazilian Portuguese Speech Dataset(BRSD):

```   
python train.py --dataset .datasets/brsd/data.h5 --model deep_speech2  --gpu all --model_params "num_features=161"


```
For English Speech Dataset (ENSD):


```   
python train.py --dataset .datasets/ensd/data.h5 --model deep_speech2  --gpu all --model_params "num_features=161"

```






