import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from urllib.parse import urlparse


CSV_PATH = os.environ.get("PHISHING_GRAPH_CSV", "graph_data.csv")
SAVE_PATH = os.environ.get("PHISHING_GRAPH_SAVE", "phishing_network_graph.png")

TOP_SENDERS = int(os.environ.get("PHISHING_TOP_SENDERS", "15"))
TOP_DOMAINS = int(os.environ.get("PHISHING_TOP_DOMAINS", "15"))

INTERNAL_DOMAINS = ["enron.com"]


def extract_domains(urls: str):
    domains = []
    for url in str(urls).split():
        try:
            domain = urlparse(url).netloc.lower()
            if domain:
                domains.append(domain)
        except Exception:
            pass
    return domains


def is_external(domain: str) -> bool:
    return not any(internal in domain for internal in INTERNAL_DOMAINS)


def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    # filter phishing only
    df = df[df["possible_phishing"] == 1].copy()
    print("Phishing emails:", len(df))
    if df.empty:
        print("[i] No phishing rows to plot.")
        return

    # domains
    df["domains"] = df["url"].apply(extract_domains)
    df["domains"] = df["domains"].apply(lambda ds: [d for d in ds if is_external(d)])

    # top nodes
    top_senders = df["from"].value_counts().head(TOP_SENDERS).index.tolist()
    top_domains = df.explode("domains")["domains"].value_counts().head(TOP_DOMAINS).index.tolist()

    print("Top senders:", len(top_senders))
    print("Top domains:", len(top_domains))

    # graph
    G = nx.DiGraph()
    for _, row in df.iterrows():
        sender = row["from"]
        if sender not in top_senders:
            continue
        for domain in row["domains"]:
            if domain in top_domains:
                G.add_edge(sender, domain)

    print("Graph nodes:", G.number_of_nodes())
    print("Graph edges:", G.number_of_edges())

    if G.number_of_nodes() == 0:
        print("[i] Graph is empty after filtering.")
        return

    centrality = nx.degree_centrality(G)

    # separate node types
    sender_nodes = [n for n in G.nodes() if "@" in n]
    domain_nodes = [n for n in G.nodes() if "@" not in n]

    pos = {}
    angle_step = 2 * np.pi / max(1, len(domain_nodes))
    radius = 4

    for i, node in enumerate(domain_nodes):
        pos[node] = (radius * np.cos(i * angle_step), radius * np.sin(i * angle_step))

    angle_step2 = 2 * np.pi / max(1, len(sender_nodes))
    for i, node in enumerate(sender_nodes):
        pos[node] = (2 * radius * np.cos(i * angle_step2), 2 * radius * np.sin(i * angle_step2))

    plt.figure(figsize=(15, 11))

    # draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=sender_nodes,
        node_color="dodgerblue",
        node_size=[15000 * centrality[n] for n in sender_nodes],
        label="Senders",
        alpha=0.85
    )

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=domain_nodes,
        node_color="crimson",
        node_size=[15000 * centrality[n] for n in domain_nodes],
        label="Domains",
        alpha=0.85
    )

    # edges
    nx.draw_networkx_edges(
        G, pos,
        edge_color="gray",
        arrows=True,
        arrowstyle="-|>",
        arrowsize=22,
        width=1.8,
        alpha=0.75
    )

    # label only top nodes
    top_nodes = sorted(centrality, key=centrality.get, reverse=True)[:25]
    labels = {n: n for n in top_nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=11)

    plt.legend(scatterpoints=1)
    plt.title("Phishing Network Graph (Top External Domains & Senders)", fontsize=18)
    plt.axis("off")

    # save + show
    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
    print(f"[✓] Saved: {SAVE_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
