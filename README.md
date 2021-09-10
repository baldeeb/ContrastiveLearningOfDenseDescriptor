# ContrastiveLearningOfDenseDescriptor


# Code TODOs:
 - [ ] Setup Proper Config system
    - [ ] Create config class
    - [ ] populate json config files
 - [ ] Clean up the dataset loader.
    - [ ] remove duplicate and unnecessary data.
    - [ ] facilitate the derivation of specific masks
 - [ ] Allow batches and assume consecutive images are augmented pairs.






# Demanding Tasks: 
- [ ] **Figure out the effect of batch size** 
    - [ ] integrate the ability to increase batch size if needed
- [x] **Matching pairs of pixels in augmented images.** _Those are necessary for the basic matching loss_
- [ ] **Better code the loss functions** _Matching loss seemed too complicated_
    - [x] Isolate essential loss components
- Investigate the possibility of using a projection layer for SimSiam like non-contrastive learning.
