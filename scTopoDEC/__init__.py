import os
# Ensure the backend is set before Keras is imported anywhere else
os.environ['KERAS_BACKEND'] = 'tensorflow'

# Bridge the function from api.py to the package level
from .api import scTopoDEC