# HFSP repo
This repository has all the codes for the HFSP project. The main branch implements the boolean model that simulates the process of bud dormancy break. The '' branch implements the information theoretic analysis of the model and the experimental data.

## HFSP model
The bud tissue is modelled as a network having cells as nodes and their connections as edges. The state of the tissue/network is the boolean state of the nodes (representing gene expression of the cell) plus the boolean state of the edges (representing PD's) The HFSP model updates the state of the network once in every hour, based on the follwoing parameters:
- p_cold (probability that a cell updates its state from 0 to 1 in the cold phase)
- p_warm (probability that a cell updates its state from 1 to 0 in the warm phase)
- p_edge (probability that an edge updates its state from 0 to 1)
- majority rule code = [{0,1,2,3}, percentage]
- edge dynamics rule code = {0,1,2}
