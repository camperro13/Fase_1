```mermaid
flowchart TD
	node1["data.csv.dvc"]
	node2["evaluate"]
	node3["load_data"]
	node4["pipeline"]
	node5["preprocess"]
	node3-->node4
	node3-->node5
	node4-->node2
	node5-->node2
	node5-->node4
```