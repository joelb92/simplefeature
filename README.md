This repository provides an incredibly light-weight implementation of the gray-scale feature extractor from:
"Beyond simple features: A large-scale feature search approach to unconstrained face recognition"

## Use

1. `pip install simplefeat` or 
   `git clone https://github.com/joelb92/simplefeature.git && cd simplefeat && python setup.py install`
2. ~~~
   git clone https://code.ornl.gov/ii1/briar-api.git`
   import simplefeature
   import cv2
   im = cv2.imread("/home/face.jpg")
   embedding = simplefeature.extract(im) 
   ~~~

Inputs will be scaled to 200x200px
The system outputs a 51200-d vector

Please cite this paper:
~~~
Cox, David, and Nicolas Pinto. "Beyond simple features: A large-scale feature search approach to unconstrained face recognition."
2011 IEEE International Conference on Automatic Face & Gesture Recognition (FG). IEEE, 2011.
~~~