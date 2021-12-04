# Learning Dense Descriptors through Contrastive Methods and Augmentations.

## Big pictures:
Zhen Zeng and Xiaotong Chen proposed and demonstrated the use of affordance coordinate frames for informing manipulation. Their work split the perception portion of the manipulation problem into two: deriving dense image descriptors, and learning affordance frames using those descriptors. Training the proposed network relies heavily on simulated data. Improving the work could be done by advancing descriptor derivation or learning the ACF vectors in a self-supervised/unsupervised manner.
## Recent Contributions: 
Xiaotong had recently proposed using the method by [Florence & Manuelli] to derive descriptors for the purpose of helping generalize affordance coordinate frame learning. Over the past couple of months I have been helping Xiaotong and Kaizhi Zheng advance their work by: 
Developing an evaluation method to assess descriptor generalizability. That method centers objects of the same category and associates their pixels then measures the distances between descriptors of those associated pixels.
Attempting (and failing) to use the intra-category association to build more general descriptors. So far this has been causing the descriptor space to collapse into a trivial solution.
Training a feature-pyramid-network as a backbone, using the same method proposed by [Florence & Manuelli]. Because that backbone is more commonly used in cases where there are objects of interest with various sizes.
Apart from that, I have been exploring literature on manipulation and perception by reading about descriptor learning, semantically informed manipulation, and learning from demonstration. A few of the papers I read inspired the proposal below.
## Learn descriptors using contrastive learning:
Recent work using contrastive learning proved that by comparing augmented views of human hands we are able to learn features that allow us to better perform hand pose estimation [Spurr & Dahiya]. The insight of this work was that geometric augmentations (e.g. rotation) should translate to a similar augmentation in feature space. Issue is that this requires iconic views (views of a single object) or at least bounding boxes on objects of interest. Other recent work has shown promising result training a network to perform object detection using non-iconic images [Liu]. Can we use contrastive learning methods to extract features that are meaningful in a manipulation pipeline, using natural non-iconic images?.
Motivation
Unsupervised dense descriptors can help generalize learned manipulation protocols (e.g. ACF) of intra category objects without having to explicitly label each instance. Current state of the art uses pixel level comparison [Florence & Manuelli] [Unsupervised Learning of Dense Visual Representations]. From some experiments we ran on the work by [Florence & Manuelli], we saw that different instances of objects of the same category do not necessarily develop similar feature distributions (that is based on simulator generated images of different mugs). The use of augmentations, that are central in contrastive learning, might encourage similarity in descriptors of objects of the same category.
Contrastive learning is easy to transfer to real world applications when it does not require object reconstruction and pixel level associations. Some current methods require 3D reconstruction of objects and performing association and dissociation, which ends up being complicated and expensive.
## Intended Exploration Directions: 
Current work could be improved by: 
Learning from non-iconic cluttered images through either using single views or possibly using a method similar to [Liu] leveraging RGB association and different resolutions. 
Using positive only samples as is done by SimSiam or SimCLR. That would allow for easier and possibly perpetual learning.
## Evident difficulties: 
The paper by [Liu] uses plenty of data to learn from non-iconic images. That said, the work does rely on association solely using RGB similarity, we might be able to improve sample efficiency by using more robust association methods leveraging camera pose and image features.
Code is not available for [Spurr & Dahiya] or [Liu] so the methods will need to be reimplemented. 
What work needs to be done?
More extensively and closely review literature and method implementations. 
Collect or identify the data needed.
Re-implement required functions for geometrically equivariant transformations.
Experiment with the use of an FPN and different resolutions to attempt to find transformations to build associations using non-iconic images (ones with many objects)**
Build a more standard evaluation method to evaluate the extent of how useful the produced features are.
Tangential experimentation: Explore the use of view augmentation for making the method presented by [Florence & Manuelli] more generalizable.
## Possible Deliverables:
The objective of this network would be to derive dense features that are sufficiently detailed to inform manipulation while being similar between instances of the same category. 
After implementing the first version of the network, I could present progress by:
- **Initial**: Visualizing the descriptors by projecting them into 3D space and manually inspecting them.
- **Viable**: Using simulated data (knowledge of object poses) and the eval tool we recently developed to compare descriptor distances between spatially close pixels of same-category objects. 
- **Viable**: Use the descriptor to train a simple detector.
- **Stretch**: Ideally, use that descriptor to run the tests used by [Florence & Manuelli] or the previous ACF paper.



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




# How to run this!

Simply run python main.py to train

to use tensorboard run: tensorboard --logdir=runs