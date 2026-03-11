"""
Fraud Ring Detection - Streamlit Demo App
Author: Venkata Nandini Mamillapalli
GitHub: https://github.com/imNandini19/
LinkedIn: https://www.linkedin.com/in/iamnandini19/

Run locally : streamlit run app.py
Deployed at : Streamlit Cloud

How this works:
  - Loads pre-saved graph_data.pkl  (adjacency list, labels, fraud rings)
  - Loads pre-saved fraud_gnn_model.pt (trained weights — no training at startup)
  - App starts in 2-3 seconds instead of 60 seconds
  - No CSV files needed at runtime
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
import pickle
import os
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
# UNION-FIND — identical to notebook
# ─────────────────────────────────────────────────────────────
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size   = [1] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
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
# ─────────────────────────────────────────────────────────────
def bfs_subgraph(adj, start_node, max_hops=2, max_nodes=40):
    visited  = set()
    queue    = deque()
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
# DETECT FRAUD RINGS — identical to notebook
# ─────────────────────────────────────────────────────────────
def detect_fraud_rings(adj, node_labels, edge_list, next_id, min_size=3):
    uf = UnionFind(next_id)
    for src, dst in edge_list:
        uf.union(src, dst)

    components  = uf.get_components()
    fraud_rings = []

    for root, members in components.items():
        if len(members) < min_size:
            continue
        labeled_members = [m for m in members if m in node_labels]
        if not labeled_members:
            continue
        fraud_in_ring = sum(1 for m in labeled_members if node_labels[m] == 0)
        fraud_ratio   = fraud_in_ring / len(labeled_members)
        if fraud_ratio >= 0.5:
            fraud_rings.append({
                "root"        : root,
                "members"     : members,
                "size"        : len(members),
                "fraud_nodes" : fraud_in_ring,
                "fraud_ratio" : fraud_ratio,
            })

    fraud_rings.sort(key=lambda x: x["size"], reverse=True)
    for i, ring in enumerate(fraud_rings):
        ring["ring_number"] = i + 1

    return fraud_rings


# ─────────────────────────────────────────────────────────────
# BUILD PyG DATA — identical to notebook
# ─────────────────────────────────────────────────────────────
def build_pyg_data(node_labels, node_features, edge_list):
    valid_nodes = sorted([n for n in node_labels if n in node_features])
    remap       = {old: new for new, old in enumerate(valid_nodes)}

    x = torch.tensor([node_features[n] for n in valid_nodes], dtype=torch.float)
    y = torch.tensor([node_labels[n]   for n in valid_nodes], dtype=torch.long)

    srcs, dsts = [], []
    for s, d in edge_list:
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
# LOAD EVERYTHING — cached, runs once
#
# Looks for graph_data.pkl + fraud_gnn_model.pt in outputs/
# If pkl is missing, falls back to building from CSVs in data/
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_everything():

    # ── Step 1: Load graph data ──────────────────────────────
    pkl_path = os.path.join("outputs", "graph_data.pkl")
    csv_path = os.path.join("data", "elliptic_txs_classes.csv")

    if os.path.exists(pkl_path):
        # Fast path — load pre-saved graph (Streamlit Cloud uses this)
        with open(pkl_path, "rb") as f:
            saved = pickle.load(f)
        node_id_map   = saved["node_id_map"]
        adj           = defaultdict(list, saved["adj"])
        node_labels   = saved["node_labels"]
        node_features = saved["node_features"]
        edge_list     = saved["edge_list"]
        fraud_rings   = saved["fraud_rings"]
        next_id       = max(node_id_map.values()) + 1

    elif os.path.exists(csv_path):
        # Slow path — build from CSVs (local machine fallback)
        from collections import defaultdict as dd

        df_cls  = pd.read_csv("data/elliptic_txs_classes.csv")
        df_edg  = pd.read_csv("data/elliptic_txs_edgelist.csv")
        df_feat = pd.read_csv("data/elliptic_txs_features.csv", header=None)

        df_cls = df_cls.rename(columns={df_cls.columns[0]: "txId", df_cls.columns[1]: "class"})
        df_edg = df_edg.rename(columns={df_edg.columns[0]: "txId1", df_edg.columns[1]: "txId2"})

        # Build graph manually (same logic as FraudGraphBuilder in notebook)
        node_id_map   = {}
        next_id       = 0
        adj           = defaultdict(list)
        node_labels   = {}
        node_features = {}
        edge_list     = []

        labeled = df_cls[df_cls["class"] != "unknown"].head(10000)
        for _, row in labeled.iterrows():
            tx = str(row["txId"])
            if tx not in node_id_map:
                node_id_map[tx] = next_id
                next_id += 1
            node_labels[node_id_map[tx]] = 0 if str(row["class"]) == "1" else 1

        node_set    = set(node_id_map.keys())
        edges_added = 0
        for _, row in df_edg.iterrows():
            s, d = str(row["txId1"]), str(row["txId2"])
            if s in node_set and d in node_set:
                si, di = node_id_map[s], node_id_map[d]
                adj[si].append(di)
                adj[di].append(si)
                edge_list.append((si, di))
                edges_added += 1
            if edges_added >= 20000:
                break

        feat_cols = df_feat.iloc[:, 1:]
        for i, row in df_feat.iterrows():
            tx = str(int(row[0]))
            if tx in node_id_map:
                ni = node_id_map[tx]
                node_features[ni] = feat_cols.iloc[i].values.tolist()

        fraud_rings = detect_fraud_rings(adj, node_labels, edge_list, next_id)

    else:
        return None, None, None, None, None, None, None, None, "MISSING"

    # ── Step 2: Build PyG data object ───────────────────────
    data, remap, valid_nodes = build_pyg_data(node_labels, node_features, edge_list)

    # ── Step 3: Load trained model weights ──────────────────
    pt_path = os.path.join("outputs", "fraud_gnn_model.pt")
    if not os.path.exists(pt_path):
        return None, None, None, None, None, None, None, None, "NO_MODEL"

    checkpoint = torch.load(pt_path, map_location="cpu")
    model      = FraudGNN(in_channels=data.num_node_features)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ── Step 4: Build node → ring lookup ────────────────────
    node_to_ring = {}
    for ring in fraud_rings:
        for member in ring["members"]:
            node_to_ring[member] = ring

    return (model, adj, node_labels, node_features,
            edge_list, data, remap, valid_nodes,
            fraud_rings, node_to_ring, node_id_map, "OK")


# ─────────────────────────────────────────────────────────────
# APP STARTS HERE
# ─────────────────────────────────────────────────────────────
with st.spinner("Loading model and graph data..."):
    result = load_everything()

# Unpack result
if result[-1] == "MISSING":
    st.error("""
    **Cannot find graph data or CSV files.**

    Run this cell in your notebook first, then push `outputs/graph_data.pkl` to GitHub:

    ```python
    import pickle
    save_data = {
        'node_id_map'   : builder.node_id_map,
        'adj'           : dict(builder.adj),
        'node_labels'   : builder.node_labels,
        'node_features' : builder.node_features,
        'edge_list'     : builder.edge_list,
        'fraud_rings'   : fraud_rings,
    }
    with open('../outputs/graph_data.pkl', 'wb') as f:
        pickle.dump(save_data, f)
    print("Done — push outputs/graph_data.pkl to GitHub")
    ```
    """)
    st.stop()

if result[-1] == "NO_MODEL":
    st.error("Cannot find `outputs/fraud_gnn_model.pt`. Push it to GitHub.")
    st.stop()

(model, adj, node_labels, node_features,
 edge_list, data, remap, valid_nodes,
 fraud_rings, node_to_ring, node_id_map, status) = result

# Reverse map: node index → txId string
reverse_map = {v: k for k, v in node_id_map.items()}

st.success(f"✅ Model loaded · {len(fraud_rings)} fraud rings detected")
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
# ════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Fraud Ring Explorer")
    st.write(
        "Pick any transaction. Union-Find tells you which fraud ring it belongs to. "
        "GraphSAGE gives the fraud probability using its 3-hop neighbourhood."
    )

    col_left, col_right = st.columns([1, 2])

    with col_left:
        node_type = st.radio(
            "Show transactions from:",
            ["🔴 Fraud nodes", "🟢 Legit nodes"],
            horizontal=True
        )
        if node_type == "🔴 Fraud nodes":
            pool = [n for n in valid_nodes if node_labels.get(n) == 0]
        else:
            pool = [n for n in valid_nodes if node_labels.get(n) == 1]

        selected_node = st.selectbox(
            "Select transaction:",
            options     = pool[:50],
            format_func = lambda n: f"txId {reverse_map.get(n, n)}"
        )

    with col_right:
        tx_id_display = reverse_map.get(selected_node, selected_node)
        true_label    = node_labels.get(selected_node)
        true_str      = "🔴 Fraud" if true_label == 0 else "🟢 Legit"
        ring          = node_to_ring.get(selected_node)

        st.markdown("#### Union-Find Result")
        if ring:
            st.success(f"This transaction belongs to **Fraud Ring #{ring['ring_number']}**")
            ring_df = pd.DataFrame([
                {"Field": "Union-Find root node",       "Value": str(ring["root"])},
                {"Field": "Ring size (total members)",  "Value": str(ring["size"])},
                {"Field": "Fraud nodes in ring",        "Value": str(ring["fraud_nodes"])},
                {"Field": "Fraud ratio",                "Value": f"{ring['fraud_ratio']:.0%}"},
                {"Field": "Source",                     "Value": "uf.find() + uf.get_components()"},
            ])
            st.dataframe(ring_df, hide_index=True, use_container_width=True)
        else:
            st.info("This transaction is not part of any detected fraud ring.")

    st.divider()

    # ── GNN Prediction ───────────────────────────────────────
    st.markdown("#### GraphSAGE Prediction")

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
        match = "✅ Correct" if prediction == true_label else "❌ Wrong"
        st.write(f"**Ground truth:** {true_str}")
        st.write(f"**Prediction:** {match}")

    with pc2:
        st.metric("Fraud Probability", f"{fraud_prob * 100:.1f}%")
        st.metric("Legit Probability", f"{legit_prob * 100:.1f}%")

    with pc3:
        neighbours       = adj.get(selected_node, [])
        fraud_neighbours = [n for n in neighbours if node_labels.get(n) == 0]
        st.metric("Direct Neighbours", len(neighbours))
        st.metric("Fraud Neighbours",  len(fraud_neighbours))

    st.divider()

    # ── BFS Neighbourhood Graph ──────────────────────────────
    st.markdown("#### 2-Hop BFS Neighbourhood")
    st.caption(
        "Yellow = selected transaction · "
        "Red = fraud · "
        "Blue = legit · "
        "Yellow border = ring members detected by Union-Find"
    )

    sub_nodes, sub_edges = bfs_subgraph(adj, selected_node, max_hops=2, max_nodes=40)
    ring_members_in_sub  = set(ring["members"]) if ring else set()
    fraud_in_sub         = [n for n in sub_nodes if node_labels.get(n) == 0]
    legit_in_sub         = [n for n in sub_nodes if node_labels.get(n) == 1]

    G   = nx.Graph()
    G.add_nodes_from(sub_nodes)
    G.add_edges_from(sub_edges)
    pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=(11, 6))
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, edge_color="gray")

    legit_only = [n for n in legit_in_sub if n != selected_node]
    if legit_only:
        nx.draw_networkx_nodes(G, pos, nodelist=legit_only, ax=ax,
                               node_color="steelblue", node_size=180, alpha=0.7)

    fraud_not_ring = [n for n in fraud_in_sub
                      if n not in ring_members_in_sub and n != selected_node]
    if fraud_not_ring:
        nx.draw_networkx_nodes(G, pos, nodelist=fraud_not_ring, ax=ax,
                               node_color="red", node_size=220, alpha=0.85)

    ring_in_sub = [n for n in ring_members_in_sub
                   if n in sub_nodes and n != selected_node]
    if ring_in_sub:
        nx.draw_networkx_nodes(G, pos, nodelist=ring_in_sub, ax=ax,
                               node_color="red", node_size=280,
                               edgecolors="yellow", linewidths=2.5)

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
                  label=f"Ring #{ring['ring_number']} members ({len(ring_in_sub)})")
        )
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9)
    ax.set_title(
        f"2-Hop BFS Neighbourhood · {len(sub_nodes)} nodes · {len(sub_edges)//2} edges",
        fontsize=11
    )
    ax.axis("off")
    st.pyplot(fig)
    plt.close()


# ════════════════════════════════════════════════════════════
# TAB 2 — ALL FRAUD RINGS
# ════════════════════════════════════════════════════════════
with tab2:
    st.subheader("All Fraud Rings Detected by Union-Find")
    st.write(
        f"Union-Find found **{len(fraud_rings)} fraud rings** in the graph. "
        f"Each row is a connected component where the majority of labeled nodes are fraudulent."
    )

    if fraud_rings:
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Total Rings Found",    len(fraud_rings))
        mc2.metric("Largest Ring",         fraud_rings[0]["size"])
        mc3.metric("Total Nodes in Rings", sum(r["size"] for r in fraud_rings))

        st.divider()

        table_rows = [{
            "Ring #"          : r["ring_number"],
            "Union-Find Root" : r["root"],
            "Total Members"   : r["size"],
            "Fraud Nodes"     : r["fraud_nodes"],
            "Fraud %"         : f"{r['fraud_ratio']:.0%}",
        } for r in fraud_rings]

        st.dataframe(pd.DataFrame(table_rows), hide_index=True,
                     use_container_width=True)

        st.caption(
            "Union-Find Root = parent node after path compression. "
            "Two nodes are in the same ring if and only if uf.find(a) == uf.find(b)."
        )
    else:
        st.info("No fraud rings detected.")


# ════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════
with tab3:
    st.subheader("GraphSAGE Model Performance on Test Set")

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
- **False Negatives (fn = {fn})** — fraud missed ← the costly error
- **False Positives (fp = {fp})** — legit flagged as fraud
- **True Negatives  (tn = {tn})** — legit correctly cleared ✅
    """)

    st.info(
        f"Recall = {recall:.2f} → catches {recall*100:.0f}% of all fraud. "
        f"More important than accuracy on imbalanced data."
    )
