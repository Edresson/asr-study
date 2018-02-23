import ast

class HParams(object):
    """Creates an object for passing around hyperparameter values.
    Use the parse method to overwrite the default hyperparameters with values
    passed in as a string representation of a Python dictionary mapping
    hyperparameters to values.

    # Example
      hparams = magenta.common.HParams(batch_size=128, hidden_size=256)
      hparams.parse('{"hidden_size":512}')
      assert hparams.batch_size == 128
      assert hparams.hidden_size == 512


      Code adpated from Google Magenta
    """

    def __init__(self, **init_hparams):
        object.__setattr__(self, 'keyvals', init_hparams)

    def __getitem__(self, key):
        """Returns value of the given hyperameter, or None if does not
        exist."""
        return self.keyvals.get(key)

    def __getattribute__(self, attribute):
        if attribute == '__dict__':
            return self.keyvals
        else:
            return object.__getattribute__(self, attribute)

    def __getattr__(self, key):
        """Returns value of the given hyperameter, or None if does not
        exist."""
        return self.keyvals.get(key)

    def __setattr__(self, key, value):
        """Sets value for the hyperameter."""
        self.keyvals[key] = value

    def update(self, values_dict):
        """Merges in new hyperparameters, replacing existing with same key."""
        self.keyvals.update(values_dict)

        return self

    def parse(self, values):
        """Merges in new hyperparameters, replacing existing with same key."""

        if type(values) == dict:
            return self.update(values)

        if type(values) in (set, list):
            try:# python2 and python3 support working 
                tmp = {}
                
                if values[0].find('=') ==  -1:# accept values ex: "num_features 10  num_classes 20"
                    
                    split = values[0].strip().split(' ')
                    
                    for i in range(0,len(split),2):
                        try:
                            tmp[split[i]] = int(split[i+1])
                        except:
                            tmp[split[i]] = float(split[i+1])
                            
                    return tmp
                else:# accept values ex: " num_features=10  num_classes=20 "
                    tmp = {}
                    split = values[0].strip().split('=')
                    aux = []
                    for i in range(len(split)):
                        lsplit  = split[i].split(' ')
                        for i in lsplit:
                            if i != '':
                                aux.append(i)
                    
                    for i in range(0,len(aux),2):
                        try:
                            tmp[aux[i]] = int(aux[i+1])#for int argument 
                        except:
                            tmp[aux[i]] = float(aux[i+1])#for float argument
                            
                        
                    return self.update(tmp)
        
            except:#python2 maintained original support
                
                tmp = {}
                
                for k, v in zip(values[::2], values[1::2]):
                    
                    try:
                        
                        tmp[k] = ast.literal_eval(v)
                    except ValueError:
                        tmp[k] = v
                
                return self.update(tmp)
        
            return self.update(ast.literal_eval(values))

    def values(self):
        """Return the hyperparameter values as a Python dictionary."""
        return self.keyvals

    def __str__(self):
        return str(self.keyvals)
