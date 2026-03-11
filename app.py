"""
Fraud Ring Detection - Streamlit Demo App
Author: Venkata Nandini Mamillapalli
GitHub: https://github.com/imNandini19/
LinkedIn: https://www.linkedin.com/in/iamnandini19/

Problem Statement:
    Traditional fraud detection looks at one transaction at a time.
    This misses coordinated fraud rings — groups of wallets that
    work together to launder money. By modeling transactions as a
    graph and using Union-Find + GraphSAGE, we detect the entire ring,
    not just individual bad actors.

Run: streamlit run app.py
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from collections import defaultdict, deque
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Fraud Ring Detector", page_icon="🔍", layout="wide")
st.title("🔍 Fraud Ring Detection — Graph Neural Network")
st.caption(
    "Elliptic Bitcoin Dataset · 203,769 transactions · "
    "Union-Find ring detection · GraphSAGE 3-layer GNN"
)

# ─────────────────────────────────────────────────────────────
# MODEL — identical to notebook
# ─────────────────────────────────────────────────────────────
class FraudGNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1      = SAGEConv(in_channels, 128)
        self.conv2      = SAGEConv(128, 64)
        self.conv3      = SAGEConv(64, 32)
        self.classifier = nn.Linear(32, 2)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        return self.classifier(x)


# ─────────────────────────────────────────────────────────────
# GRAPH BUILDER — identical to notebook
# ─────────────────────────────────────────────────────────────
class FraudGraphBuilder:
    def __init__(self):
        self.node_id_map   = {}                 # hash map: string txId → integer index
        self.next_id       = 0
        self.adj           = defaultdict(list)  # adjacency list: node → [neighbours]
        self.node_labels   = {}                 # node → 0 (fraud) or 1 (legit)
        self.node_features = {}                 # node → 166 features
        self.edge_list     = []

    def get_or_create_node(self, tx_id):
        tx_id = str(tx_id)
        if tx_id not in self.node_id_map:
            self.node_id_map[tx_id] = self.next_id
            self.next_id += 1
        return self.node_id_map[tx_id]

    def build(self, df_classes, df_edges, df_features, max_nodes=10000):
        labeled = df_classes[df_classes['class'] != 'unknown'].head(max_nodes)
        for _, row in labeled.iterrows():
            idx = self.get_or_create_node(row['txId'])
            self.node_labels[idx] = 0 if str(row['class']) == '1' else 1

        node_set    = set(self.node_id_map.keys())
        edges_added = 0
        for _, row in df_edges.iterrows():
            s, d = str(row['txId1']), str(row['txId2'])
            if s in node_set and d in node_set:
                si, di = self.node_id_map[s], self.node_id_map[d]
                self.adj[si].append(di)
                self.adj[di].append(si)
                self.edge_list.append((si, di))
                edges_added += 1
            if edges_added >= max_nodes * 2:
                break

        feat_cols = df_features.iloc[:, 1:]
        for i, row in df_features.iterrows():
            tx_id = str(int(row[0]))
            if tx_id in self.node_id_map:
                ni = self.node_id_map[tx_id]
                self.node_features[ni] = feat_cols.iloc[i].values.tolist()
        return self


# ─────────────────────────────────────────────────────────────
# UNION-FIND — identical to notebook
# This is what detects fraud rings.
# find()  → which ring does this node belong to?
# union() → merge two nodes into the same ring
# get_components() → return all rings as { root: [members] }
# ─────────────────────────────────────────────────────────────
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size   = [1] * n

    def find(self, x):
        # Path compression — every node points directly to root after first call
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        # Union by size — attach smaller tree under larger tree
        if self.size[rx] < self.size[ry]:
            rx, ry = ry, rx
        self.parent[ry]  = rx
        self.size[rx]   += self.size[ry]

    def get_components(self):
        components = defaultdict(list)
        for node in range(len(self.parent)):
            components[self.find(node)].append(node)
        return dict(components)


# ─────────────────────────────────────────────────────────────
# BFS — identical to notebook
# Extracts k-hop neighbourhood around a node.
# Used to visualise the ring structure in the graph.
# ─────────────────────────────────────────────────────────────
def bfs_subgraph(adj, start_node, max_hops=2, max_nodes=40):
    visited  = set()
    queue    = deque()   # deque not list — popleft is O(1) not O(n)
    hop_dist = {}

    queue.append(start_node)
    visited.add(start_node)
    hop_dist[start_node] = 0

    while queue:
        current = queue.popleft()
        dist    = hop_dist[current]
        if dist >= max_hops or len(visited) >= max_nodes:
            continue
        for neighbor in adj.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                hop_dist[neighbor] = dist + 1
                queue.append(neighbor)

    sub_edges = [
        (n, nb) for n in visited
        for nb in adj.get(n, []) if nb in visited
    ]
    return visited, sub_edges


# ─────────────────────────────────────────────────────────────
# BUILD PyG DATA — identical to notebook
# ─────────────────────────────────────────────────────────────
def build_pyg_data(builder):
    valid_nodes = sorted([n for n in builder.node_labels if n in builder.node_features])
    remap       = {old: new for new, old in enumerate(valid_nodes)}

    x = torch.tensor([builder.node_features[n] for n in valid_nodes], dtype=torch.float)
    y = torch.tensor([builder.node_labels[n]   for n in valid_nodes], dtype=torch.long)

    srcs, dsts = [], []
    for s, d in builder.edge_list:
        if s in remap and d in remap:
            srcs += [remap[s], remap[d]]
            dsts += [remap[d], remap[s]]
    edge_index = torch.tensor([srcs, dsts], dtype=torch.long)

    N    = len(valid_nodes)
    perm = torch.randperm(N)
    cut  = int(0.8 * N)
    train_mask = torch.zeros(N, dtype=torch.bool)
    test_mask  = torch.zeros(N, dtype=torch.bool)
    train_mask[perm[:cut]] = True
    test_mask[perm[cut:]]  = True

    return Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, test_mask=test_mask), remap, valid_nodes


# ─────────────────────────────────────────────────────────────
# DETECT FRAUD RINGS using Union-Find
# Returns list of rings sorted by size (largest first)
# ─────────────────────────────────────────────────────────────
def detect_fraud_rings(builder, min_size=3):
    """
    Run Union-Find on all edges.
    A ring = connected component where majority of labeled nodes are fraud.
    Returns a list of dicts, one per ring, sorted by size descending.
    """
    uf = UnionFind(builder.next_id)

    # Union every edge — this builds the connected components
    for src, dst in builder.edge_list:
        uf.union(src, dst)

    components  = uf.get_components()
    fraud_rings = []

    for root, members in components.items():
        if len(members) < min_size:
            continue

        # Only count labeled nodes when computing fraud ratio
        labeled_members = [m for m in members if m in builder.node_labels]
        if not labeled_members:
            continue

        fraud_in_ring = sum(1 for m in labeled_members if builder.node_labels[m] == 0)
        fraud_ratio   = fraud_in_ring / len(labeled_members)

        if fraud_ratio >= 0.5:
            fraud_rings.append({
                "root"        : root,           # Union-Find root — proves origin
                "members"     : members,        # all nodes in this component
                "size"        : len(members),
                "fraud_nodes" : fraud_in_ring,
                "fraud_ratio" : fraud_ratio,
            })

    fraud_rings.sort(key=lambda x: x["size"], reverse=True)

    # Assign readable ring numbers AFTER sorting by size
    for i, ring in enumerate(fraud_rings):
        ring["ring_number"] = i + 1

    return fraud_rings, uf


# ─────────────────────────────────────────────────────────────
# LOAD + TRAIN — cached, runs once on first launch
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_and_train():
    try:
        df_cls  = pd.read_csv('data/elliptic_txs_classes.csv')
        df_edg  = pd.read_csv('data/elliptic_txs_edgelist.csv')
        df_feat = pd.read_csv('data/elliptic_txs_features.csv', header=None)
    except FileNotFoundError:
        return None, None, None, None, None, None, None, "MISSING"

    df_cls = df_cls.rename(columns={df_cls.columns[0]: 'txId',  df_cls.columns[1]: 'class'})
    df_edg = df_edg.rename(columns={df_edg.columns[0]: 'txId1', df_edg.columns[1]: 'txId2'})

    # Build graph
    builder = FraudGraphBuilder()
    builder.build(df_cls, df_edg, df_feat, max_nodes=10000)

    # Detect all fraud rings up front — shared by all tabs
    fraud_rings, uf = detect_fraud_rings(builder, min_size=3)

    # Build a lookup: node → which ring it belongs to (if any)
    node_to_ring = {}
    for ring in fraud_rings:
        for member in ring["members"]:
            node_to_ring[member] = ring

    # Build PyG data and train model
    data, remap, valid_nodes = build_pyg_data(builder)

    fc    = data.y[data.train_mask].eq(0).sum().item()
    lc    = data.y[data.train_mask].eq(1).sum().item()
    total = fc + lc
    cw    = torch.tensor([total / (2 * fc), total / (2 * lc)], dtype=torch.float)

    model     = FraudGNN(data.num_node_features)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out  = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    return model, builder, data, remap, valid_nodes, fraud_rings, node_to_ring, "OK"


# ─────────────────────────────────────────────────────────────
# APP STARTS HERE
# ─────────────────────────────────────────────────────────────
with st.spinner("Loading data, detecting fraud rings, training model... (~60 sec first time)"):
    model, builder, data, remap, valid_nodes, fraud_rings, node_to_ring, status = load_and_train()

if status == "MISSING":
    st.error("Put the 3 Elliptic CSV files inside a `data/` folder next to app.py")
    st.stop()

# Reverse map: builder node index → txId string (for display)
reverse_map = {v: k for k, v in builder.node_id_map.items()}

st.success(f"✅ Ready — {len(fraud_rings)} fraud rings detected in the graph")
st.divider()


# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🕵️ Fraud Ring Explorer",
    "📋 All Fraud Rings",
    "📊 Model Performance",
])


# ════════════════════════════════════════════════════════════
# TAB 1 — FRAUD RING EXPLORER
# This is the main demo tab. Core of the problem statement.
# Pick any transaction → see its ring → see the GNN prediction
# ════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Fraud Ring Explorer")
    st.write(
        "Pick any transaction. Union-Find tells you which fraud ring it belongs to. "
        "GraphSAGE gives the fraud probability using its 3-hop neighbourhood."
    )

    # Let user pick from fraud nodes so they always see a ring
    col_left, col_right = st.columns([1, 2])

    with col_left:
        node_type = st.radio(
            "Show transactions from:",
            ["🔴 Fraud nodes", "🟢 Legit nodes"],
            horizontal=True
        )

        if node_type == "🔴 Fraud nodes":
            pool = [n for n in valid_nodes if builder.node_labels.get(n) == 0]
        else:
            pool = [n for n in valid_nodes if builder.node_labels.get(n) == 1]

        # Show actual txId in the dropdown
        selected_node = st.selectbox(
            "Select transaction:",
            options     = pool[:50],
            format_func = lambda n: f"txId {reverse_map.get(n, n)}"
        )

        st.button("Explore →", type="primary", key="explore_btn")

    # Always show results (no button gate — makes demo smoother)
    with col_right:
        tx_id_display = reverse_map.get(selected_node, selected_node)
        true_label    = builder.node_labels.get(selected_node)
        true_str      = "🔴 Fraud" if true_label == 0 else "🟢 Legit"

        # ── Union-Find ring info ─────────────────────────────
        ring = node_to_ring.get(selected_node)

        st.markdown("#### Union-Find Result")
        if ring:
            uf_root = ring["root"]
            st.success(f"This transaction belongs to **Fraud Ring #{ring['ring_number']}**")

            # Show exactly what Union-Find produced — this is the proof
            ring_df = pd.DataFrame([{
                "Field"  : "Union-Find root node",
                "Value"  : str(uf_root),
            }, {
                "Field"  : "Ring size (total members)",
                "Value"  : str(ring["size"]),
            }, {
                "Field"  : "Fraud nodes in ring",
                "Value"  : str(ring["fraud_nodes"]),
            }, {
                "Field"  : "Fraud ratio",
                "Value"  : f"{ring['fraud_ratio']:.0%}",
            }, {
                "Field"  : "Source",
                "Value"  : "uf.find() + uf.get_components()",
            }])
            st.dataframe(ring_df, hide_index=True, use_container_width=True)

        else:
            st.info("This transaction is not part of any detected fraud ring.")

    st.divider()

    # ── GNN Prediction ───────────────────────────────────────
    st.markdown("#### GraphSAGE Prediction")

    model.eval()
    with torch.no_grad():
        out   = model(data.x, data.edge_index)
        probs = torch.softmax(out, dim=1)

    pyg_idx    = remap[selected_node]
    fraud_prob = probs[pyg_idx, 0].item()
    legit_prob = probs[pyg_idx, 1].item()
    prediction = 0 if fraud_prob > 0.5 else 1

    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        if prediction == 0:
            st.error("🚨 GNN says: FRAUD")
        else:
            st.success("✅ GNN says: LEGITIMATE")
        st.write(f"**Ground truth:** {true_str}")
        match = "✅ Correct" if prediction == true_label else "❌ Wrong"
        st.write(f"**Prediction:** {match}")

    with pc2:
        st.metric("Fraud Probability", f"{fraud_prob * 100:.1f}%")
        st.metric("Legit Probability", f"{legit_prob * 100:.1f}%")

    with pc3:
        neighbours       = builder.adj.get(selected_node, [])
        fraud_neighbours = [n for n in neighbours if builder.node_labels.get(n) == 0]
        st.metric("Direct Neighbours",  len(neighbours))
        st.metric("Fraud Neighbours",   len(fraud_neighbours))

    st.divider()

    # ── BFS Neighbourhood Graph ──────────────────────────────
    st.markdown("#### 2-Hop BFS Neighbourhood (what GNN used to predict)")
    st.caption(
        "Yellow = selected transaction · "
        "Red = fraud nodes · "
        "Blue = legit nodes · "
        "Red border = ring members detected by Union-Find"
    )

    sub_nodes, sub_edges = bfs_subgraph(
        builder.adj, selected_node, max_hops=2, max_nodes=40
    )

    # Ring members that appear in the subgraph
    ring_members_in_sub = set(ring["members"]) if ring else set()

    fraud_in_sub = [n for n in sub_nodes if builder.node_labels.get(n) == 0]
    legit_in_sub = [n for n in sub_nodes if builder.node_labels.get(n) == 1]

    G   = nx.Graph()
    G.add_nodes_from(sub_nodes)
    G.add_edges_from(sub_edges)
    pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=(11, 6))

    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, edge_color="gray")

    # Draw legit nodes
    if legit_in_sub:
        nx.draw_networkx_nodes(G, pos, nodelist=legit_in_sub, ax=ax,
                               node_color="steelblue", node_size=180, alpha=0.7)

    # Draw fraud nodes (not ring members)
    fraud_not_ring = [n for n in fraud_in_sub
                      if n not in ring_members_in_sub and n != selected_node]
    if fraud_not_ring:
        nx.draw_networkx_nodes(G, pos, nodelist=fraud_not_ring, ax=ax,
                               node_color="red", node_size=220, alpha=0.85)

    # Draw ring members — highlighted with thick border so you can see the ring
    ring_in_sub = [n for n in ring_members_in_sub
                   if n in sub_nodes and n != selected_node]
    if ring_in_sub:
        nx.draw_networkx_nodes(G, pos, nodelist=ring_in_sub, ax=ax,
                               node_color="red", node_size=280,
                               edgecolors="yellow", linewidths=2.5)

    # Draw the selected node — biggest, most visible
    nx.draw_networkx_nodes(G, pos, nodelist=[selected_node], ax=ax,
                           node_color="yellow", node_size=450,
                           edgecolors="black", linewidths=2)

    legend_handles = [
        Patch(color="yellow",    label=f"Selected: txId {tx_id_display}"),
        Patch(color="red",       label=f"Fraud nodes ({len(fraud_in_sub)})"),
        Patch(color="steelblue", label=f"Legit nodes ({len(legit_in_sub)})"),
    ]
    if ring_in_sub:
        legend_handles.append(
            Patch(facecolor="red", edgecolor="yellow", linewidth=2,
                  label=f"Ring #{ring['ring_number']} members in view ({len(ring_in_sub)})")
        )
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9)
    ax.set_title(
        f"2-Hop BFS Neighbourhood · {len(sub_nodes)} nodes · {len(sub_edges)//2} edges",
        fontsize=11
    )
    ax.axis("off")
    st.pyplot(fig)
    plt.close()

    st.caption(
        f"Nodes with yellow border = confirmed ring members from Union-Find. "
        f"The GNN aggregated all {len(sub_nodes)} nodes in this view to make its prediction."
    )


# ════════════════════════════════════════════════════════════
# TAB 2 — ALL FRAUD RINGS
# Simple table showing every ring Union-Find detected.
# Proves the algorithm found real coordinated fraud clusters.
# ════════════════════════════════════════════════════════════
with tab2:
    st.subheader("All Fraud Rings Detected by Union-Find")
    st.write(
        f"Union-Find found **{len(fraud_rings)} fraud rings** in the graph. "
        f"Each row is a connected component where the majority of labeled nodes are fraudulent. "
        f"The Union-Find root is the representative node for that component after path compression."
    )

    if fraud_rings:
        # Summary metrics
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Total Rings Found",     len(fraud_rings))
        mc2.metric("Largest Ring",          fraud_rings[0]["size"])
        mc3.metric("Total Nodes in Rings",  sum(r["size"] for r in fraud_rings))

        st.divider()

        # Full table — every column comes directly from Union-Find output
        table_rows = []
        for ring in fraud_rings:
            table_rows.append({
                "Ring #"              : ring["ring_number"],
                "Union-Find Root"     : ring["root"],       # ← direct from uf.find()
                "Total Members"       : ring["size"],
                "Fraud Nodes"         : ring["fraud_nodes"],
                "Fraud %"             : f"{ring['fraud_ratio']:.0%}",
            })

        st.dataframe(
            pd.DataFrame(table_rows),
            hide_index       = True,
            use_container_width = True
        )

        st.caption(
            "Union-Find Root = the parent node after path compression. "
            "Two nodes share a root if and only if they are in the same connected component. "
            "This is the direct output of uf.find() — no post-processing."
        )

    else:
        st.info("No fraud rings detected. Try lowering the minimum size in the code.")


# ════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════
with tab3:
    st.subheader("GraphSAGE Model Performance on Test Set")

    model.eval()
    with torch.no_grad():
        out  = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)

    y_true = data.y[data.test_mask]
    y_pred = pred[data.test_mask]

    tp = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 0) & (y_true == 1)).sum())
    fn = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 1) & (y_true == 1)).sum())

    accuracy  = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy",  f"{accuracy:.4f}")
    col2.metric("F1 Score",  f"{f1:.4f}")
    col3.metric("Precision", f"{precision:.4f}")
    col4.metric("Recall",    f"{recall:.4f}")

    st.divider()

    st.write("**Confusion Matrix:**")
    cm_df = pd.DataFrame(
        [[tp, fn], [fp, tn]],
        index   = ["Actual Fraud", "Actual Legit"],
        columns = ["Predicted Fraud", "Predicted Legit"]
    )
    st.dataframe(cm_df, use_container_width=True)

    st.write(f"""
- **True Positives  (tp = {tp})** — fraud caught correctly ✅
- **False Negatives (fn = {fn})** — fraud missed ← the costly error in real fraud detection
- **False Positives (fp = {fp})** — legit flagged as fraud (goes to human review)
- **True Negatives  (tn = {tn})** — legit correctly cleared ✅
    """)

    st.info(
        f"**Why recall ({recall:.2f}) matters most:** "
        f"Missing a fraud case costs far more than a false alarm. "
        f"Our model catches {recall*100:.0f}% of all fraud cases in the test set."
    )
