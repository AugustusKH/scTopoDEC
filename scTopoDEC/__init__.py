import os
# Ensure the backend is set before Keras is imported anywhere else
os.environ['KERAS_BACKEND'] = 'tensorflow'

# Add version
__version__ = "0.1.0"

# Bridge the function from api.py to the package level
from .api import scTopoDEC