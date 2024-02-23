# MAGIC

MAGIC is a model agnostic efficient inference method for deriving node embedding on an updated graph.

## Application Scenario

When a graph learning model is trained on the original graph $G_t$, nodes on $G_{t}$ is equipped with node embedding. When $G_{t}$ is updated to $G_{t+1}$, we can use MAGIC to efficiently generate embedding for nodes on $G_{t+1}$. 

## Magic Method

For a new node v on updated graph $G_{t+1}=(V_t,E_t)$, we select its 1-hop neighbors on the original graph $G_{t}=(V_{t+1},E_{t+1})$ and aggregated these nodes embedding. The aggreagted embedding is combined with v's node feature and fed into an MLP predictor. The predictor needs to be adapted to the updated graph by several steps of learning. 

## Implement Details

All code is written in Python 3.10.12, utilizing PyTorch 2.1.0+cu121.  We demonstrated the inference capabilities of MAGIC based on the PyG (PyTorch Geometric) framework, using the Cora dataset as an example. The dataset is divided into 3 parts. The first part is the original graph, a graph learning model is trained on it and we have embedding for nodes on the original graph. In our code, we choose GCN as the graph learning model on the original graph and the original graph contains 25% of nodes of the full graph.
