# MAGIC

MAGIC is a model agnostic efficient inference method for deriving node embedding on an updated graph.

## Application Scenario

When a graph learning model is trained on the original graph $G_t$, nodes on $G_{t}$ is equipped with node embedding. When $G_{t}$ is updated to $G_{t+1}$, we can use MAGIC to generate embedding for nodes on $G_{t+1}$. 

## Magic Method

For a new node v on updated graph $G_{t+1}$, we select its 1-hop neighbors on the original graph $G_{t}$ and aggregated these nodes embedding. The aggreagted embedding is combined with v's node feature and fed into an MLP, 
