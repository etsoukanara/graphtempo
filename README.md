# GraphTempo
The code repository for the following paper:

## GraphTempo: An aggregation framework for evolving graphs

Evangelia Tsoukanara, Georgia Koloniari, and Evaggelia Pitoura. GraphTempo: An aggregation framework for evolving graphs.

Paper accepted at the 26th International Conference on Extending Database Technology (EDBT 2023)

## Abstract
> Graphs offer a generic abstraction for modeling entities and the
> interactions and relationships between them. Since most real-
> world graphs evolve over time, there is a need for models to
> explore the evolution of graphs over time. In this paper, we intro-
> duce the GraphTempo model that allows aggregation both at the
> attribute and at the time dimension.We also propose an explo-
> ration strategy for navigating through the evolution of the graph
> based on identifying time intervals of significant growth, shrink-
> age or stability.

## General Information
This repository facilitates both temporal and attribute aggregation by introducing a set of temporal projections and enabling aggregation on static, time-varying node attributes and combinations of them. Also, provide an exploratory strategy to assess the evolution of the graph in terms of _stability_, _shrinkage_, and _growth_. The datasets used in this paper is provided in `datasets`.

## Datasets
_DBLP_: directed collaboration dataset that spans over a period of 21 years (2000 to 2020) and includes publicatoins at 21 conferences related to data management research areas. Each node corresponds to an author and is attributed with one static label (gender), and a time-varying one (#publications)

_MovieLens_: directed mutual rating dataset (built on the benchmark movie ratings dataset) covering a period of six months (May 1st, 2000 to October 31st, 2000) where each node represents a user and an edge denotes that two users have rated the same movie, and is attributed with three static (gender, age, occupation) and one time-varying attribute (average rating per month)

## Dependencies
Python 3.7

## Reproducing the results
To run the aggregation algorithms, you need to run graphtempo.py inside the `datasets` folder, and provide a set of arguments to customize the aggregation, that is: 1. the folder of the dataset, 2. the type of temporal operator (union | intersection | difference), 3. the type of attributes (static | time-varying | mixed), 4. the type of aggregation (all | distinct), 5. the preferred static attributes (gender for DBLP, and gender, age, occupation for MovieLens dataset) or any combination of them, 6. the preferred time-varying attributes (#publications for DBLP, and rating for MovieLens), 7. starting time point for the first interval, 8. ending time point for the first interval, 9. starting time point for the second interval, 10. ending time point for the second interval.

    Parameters
    1. dblp_dataset | movielens_dataset
    2. 2. u | x | f
    3. s | v | m
    4. a | d
    5. g | a | o | ga | go | ao | gao
    6. p | r
    7., 8., 9., 10. index (0 corresponds to the 1st time point)

    Example:
    ./graphtempo.py movielens_dataset u v d g p 0 2 5 6

Outputs a .txt file with the aggregate nodes information (attribute: weight) and a .txt file with the aggregate edges information ((attribute,attribute): weight).


To run the exploration algorithms, you need to run exploration.py inside the `datasets` folder, and provide the dataset folder.

    Example:
    ./exploration.py dblp_dataset

Outputs a .txt file for each event (stability, growth, shrinkage) with the intervals corresponding for each _k_ as defined in our **Evaluation** section.
