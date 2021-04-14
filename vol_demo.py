

# %%
import os.path as op 
import sys 
import numpy as np
from svb.main import run
from svb.data import SurfaceModel, VolumetricModel

import sys 
import os.path as op
sys.path.append(op.join(op.dirname(__file__), '../svb_models_asl'))

from svb_models_asl import AslRestModel 


try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except ImportError:
    import tensorflow as tf

# To make tests repeatable
tf.set_random_seed(1)
np.random.seed(1)

plds = [2.0]
repeats = 10
CBF = 60 
ATT = 1.3
TAU = 1.0
NOISE_STD = 7
size = 5 * np.ones(3, dtype=int)

data = np.zeros((*size, len(plds) * repeats))
asl_model = AslRestModel(
        VolumetricModel(data), 
        plds=plds, casl=True, repeats=repeats, tau=TAU, att=ATT)

tpts = asl_model.tpts()
with tf.Session() as sess:
    ones = np.ones([size.prod(), 1], dtype=np.float32)
    data = sess.run(asl_model.evaluate([CBF * ones], tpts))
data = data.reshape(*size, tpts.shape[-1])
data += np.random.normal(0, NOISE_STD, size=data.shape)

options = {
    "learning_rate" : 0.1,
    "batch_size" : 5,
    "sample_size" : 5,
    "epochs" : 3000,
    "log_stream" : sys.stdout,
    "display_step": 10,  
    
    "plds": plds, 
    "repeats": repeats, 
    "casl": True, 
    "tau": TAU,
    "prior_type": "M",

    # "infer_ak": False, 
    # "ak": 10, 

    # "param_overrides" : {
    #     "ftiss" : {
    #         "prior_type" : "M",
    #     },
    # }
}

runtime, svb, training_history = run(
    data, "aslrest",
    "vol_demo", 
    **options)

# epoch] = noise_params[:,0]
#             # aks = []
#             # for idx, prior in enumerate(self.prior.priors):
#             #     try:
#             #         ak = np.exp(self.evaluate(prior.logak))
#             #     except:
#             #         ak = 0
#             #     training_history["ak"][epoch,idx] = ak
#             #     aks.append(ak)
#             # aks = np.array(aks)
#             aks = np.array([self.evaluate("ak")])
