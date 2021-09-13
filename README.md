# ContrastiveLearningOfDenseDescriptor



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




