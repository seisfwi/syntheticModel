"""
syntheticModel
"""

from syntheticModel.syntheticModel import geoModel
from syntheticModel.deposit import basic as deposit
from syntheticModel.erodeRiver import basic as erodeRiver
from syntheticModel.erodeFlat import basic as erodeFlat
from syntheticModel.fault import basic as fault
from syntheticModel.salt import basic as salt
from syntheticModel.squish import basic as squish
from syntheticModel.data import *
from syntheticModel.Hypercube import *

# added by Haipeng
# Need to install jtk to use the following modules
# from syntheticModel.chimney import big as chimneyBig
# from syntheticModel.chimney import small as chimneySmall
# from syntheticModel.structuralSmoothing import smoother as structuralSmoothing


# if "ON"=="@USE_SEP@":
#     from Hypercube import *
#     from syntheticModel.chimney import big as chimneyBig
#     from syntheticModel.chimney import small as chimneySmall
#     from syntheticModel.structuralSmoothing import smoother as structuralSmoothing
# elif "OFF"=="@USE_SEP@":
#     from syntheticModel.Hypercube import *    
#     from syntheticModel.chimney import big as chimneyBig
#     from syntheticModel.chimney import small as chimneySmall
#     from syntheticModel.structuralSmoothing import smoother as structuralSmoothing
# else:
#     from syntheticModel.Hypercube import *