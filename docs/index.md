###Available models: 
####Graves2006:
see the reference at:[Graves' model] (ftp://ftp.idsia.ch/pub/juergen/icml2006.pdf)

####eyben:
see the reference at:[Eybens' model] (ftp://ftp.idsia.ch/pub/juergen/icml2006.pdf)

####maas:
see the reference at:[Maas' model] (ftp://ftp.idsia.ch/pub/juergen/icml2006.pdf)


####deep_speech:
see the reference at:[Deep Speech model] (ftp://ftp.idsia.ch/pub/juergen/icml2006.pdf)


####deep_speech2
see the reference at:[Deep Speech2 model] (ftp://ftp.idsia.ch/pub/juergen/icml2006.pdf)

####brsmv1
BRSM v1.0 model


## Example of training for all available models:



### Graves 2006
This model is trained using MFCC, using 26 coefficients, also does not use Delta and Delta-Delta.

You must preprocess the dataset in an hdf5 file by using the following parameters :

For Brazilian Portuguese Speech Dataset(BRSD) Click [here](docs/datasets.md) for more information:
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






