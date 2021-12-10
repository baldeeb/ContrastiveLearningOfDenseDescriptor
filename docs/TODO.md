
# Tasks:
## Base Tasks: 
 - [x] Setup contrastive loss
 - [x] Setup Augmentation tools:
    - [x] Add inverse capability
    - [x] Add resize capability
 - [x] Add sampling tools
 - [x] Setup main training loop
 - [ ] Setup Proper Config system
 - [ ] Clean-up/Update dataloader
 - [ ] Allow different batch sizes _(Currently limited by compute capabilities)_
 - [ ] Will using target object segmentation help?
    - Motivation: Prior dense descriptors used masks to sample.
    - [x] Implemented depth clustering functionality 
    - [ ] _(in progress)_ Used segmentation maps from data
        - That did not yield much improvement but bugs might have been the issue

 
## Demanding Tasks: 
- [ ] **Advance/experiment-with loss**
    - [ ] IMPORTANT: fix divergence loss.
    - [ ] What is the effect of using NCE loss instead?
    - [ ] Pyramidal Loss: train the network at different hierarchies
        - [x] update the geometric inverse of augmentations to allow for various sizes
        - [x] setup training class
        - [ ] figure out how to sample from different levels
            - [x] consolidate sampling and loss related functions
            - [ ] apply sampling and loss on differently sized images
            - [ ] set up iterative loop to 
- [ ] Investigate the possibility of using a projection layer for SimSiam like non-contrastive learning.
    - [ ] select box around object and use it to derive a descriptor. Descriptors from different augmentations can be pushed together.

- [ ] **Figure out the effect of batch size** 
    - [ ] integrate the ability to increase batch size if needed
- [x] **Matching pairs of pixels in augmented images.** _Those are necessary for the basic matching loss_
- [x] **Better code the loss functions** _Matching loss seemed too complicated_
    - [x] Isolate relevant loss components.


