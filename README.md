# ContrastiveLearningOfDenseDescriptor


# NEW TASKS: 
- [ ] review README file
    - [ ] update it to contain a description of the project proposal as well as current results.
    - [ ] clean up the todo list

- [ ] Pyramidal Loss: train the network at different hierarchies
    - [x] update the geometric inverse of augmentations to allow for various sizes
    - [x] setup training class
    - [ ] figure out how to sample from different levels
        - [x] consolidate sampling and loss related functions
        - [ ] apply sampling and loss on differently sized images
        - [ ] set up iterative loop to 

# Demanding Tasks: 
- [ ] **Advance/experiment-with loss**
    - [ ] IMPORTANT: fix divergence loss.
    - [ ] What is the effect of using NCE loss instead?

- [ ] **Figure out the effect of batch size** 
    - [ ] integrate the ability to increase batch size if needed
- [x] **Matching pairs of pixels in augmented images.** _Those are necessary for the basic matching loss_
- [ ] **Better code the loss functions** _Matching loss seemed too complicated_
    - [x] Isolate relevant loss components.


# Major directions:
- [ ] Investigate the possibility of using a projection layer for SimSiam like non-contrastive learning.
    - [ ] select box around object and use it to derive a descriptor. Descriptors from different augmentations can be pushed together.



# Code TODOs:
 - [ ] Setup Proper Config system
    - [ ] Create config class
    - [ ] populate json config files
 - [ ] Clean up the dataset loader.
    - [ ] remove duplicate and unnecessary data.
    - [ ] facilitate the derivation of specific masks
 - [ ] Allow batches and assume consecutive images are augmented pairs.
 - [ ] Revise Gaussian blur kernel in the augmentation




# How to run this!

Simply run python main.py to train

to use tensorboard run: tensorboard --logdir=runs