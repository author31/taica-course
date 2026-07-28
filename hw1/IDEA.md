# What is this documents used for?
Me, the author, of this repository is trying to outline the ideas and thoughts behind the HW1 of the TAICA course.
You, as an assistant, are supposed to read this document and use the information as the context for latter discussions.
You would help me by providing feedbacks and suggestions on the ideas and thoughts. refine to a concrete CLAUDE.md and the implementation plan.

# Primary Package Manager
We are going to use [Pixi](https://github.com/prefix-dev/pixi/) as the primary package manager.

## Reason
- Compatible with both conda and pip.
- Compatible with all major operating systems: Linux, Windows, macOS (including Apple Silicon)


## Primary simulator
- Habitat Sim: https://github.com/facebookresearch/habitat-sim
- Habitat Lab: https://github.com/facebookresearch/habitat-lab

# Goal of the homework 1 assignment
The goal of this homework is training students to not only understand the algorithms implementation but also to reason the success/failure algorithms perforamnce. As in the physical world, the algorithm is only puzzle of the whole system, to deploy it, there are plenty factors that could affect the algorithm's performance. Students should be well-prepared with critical thinking, analysis skills to deal with these issues. 

To integrate such workflow into the homework, ontology, semantic layers are utilized. We, as a homework designer, are going to provide definitions for the data quality that may affect the performance of the geometric ICP algorithm.

## Dataset structure
For the dataset, its structure is as follows:
- rgb/: contains the rgb images of the corrupted dataset
- depth/: contains the depth images of the corrupted dataset
- semantic/: contains the semantic labels of the corrupted dataset

## Data quality definitions
We are going to provide definitions for the data quality that may affect the performance of the geometric ICP algorithm. 

### Depth frames
- valid depth ratio: the ratio of valid depth pixels to total depth pixels
- edge density: gradient of the depth image (USER THOUGHT: help me confirm this definition, ensure this is literature grounded)
USER_THOUGHT: what are other possible gradient analyses that could help to define the quality of the depth image?

### RGB frames
USER_THOUGHT: what are the other possible quality metrics that could help to define the quality of the RGB image?

## Simluation Environment
- Dataset: "apartment_0"
    Contains two floors:
    - First floor
    - Second floor

## Details
In this homework, we'll choose geometric ICP algorithm as the algorithm to be implemented. Students are asked to go through two phases of the implementation:
1. Students deal with first floor corrupted dataset provided by us:
In this phase, students are asked to do the following:
- implement the algorithm
- implement the ontology APIs based on the definitions provided above
- load the dataset into the triple store, the Fuseki server
- reason about the performance of the algorithm, either the success or failure of the algorithm using those APIs. 
- provide a detailed report of the reasoning process, including the results of the reasoning and the analysis of the data quality.

2. Students collect their own dataset on the second floor
- Students control the agent in the simulator, and collect the data from the agent by navigating to the second floor
- Students are asked to do the following:
    - load the dataset into the triple store, the Fuseki server
    - reason about the performance of the algorithm, either the success or failure of the algorithm using those APIs.
    - provide a detailed report of the reasoning process, including the results of the reasoning and the analysis of the data quality.

