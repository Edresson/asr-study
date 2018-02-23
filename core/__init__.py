import sys
# compatibility  python2 and python3 
if sys.version_info[0] == 2:
    import layers
    import layers_utils
    import metrics
    import ctc_utils
    import models
    import initializers
    import callbacks
else:
    import core.layers
    import core.layers_utils
    import core.metrics
    import core.ctc_utils
    import core.models
    import core.initializers
    import core.callbacks
    
