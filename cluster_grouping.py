clusters = np.unique(labels)
cluster_masks = []
cluster_lkhd = []

for cluster in clusters:
    cluster_masks.append(labels == cluster)
    cluster_lkhd.append(tree._nodes_dict[cluster].lkhd)

cluster_count = len(cluster_masks)

while cluster_count > 16:
    degradation_matrix = np.full(shape=(cluster_count, cluster_count), fill_value=-np.inf)
    for i in range(cluster_count):
        for j in range(cluster_count):
            if i >= j:
                continue

            mask = (cluster_masks[i] | cluster_masks[j])
            joint_lkhd = _batch_log_lkhd(n[mask], s[mask])
            lkhd_degradation = joint_lkhd - cluster_lkhd[i] - cluster_lkhd[j]

            degradation_matrix[i, j] = lkhd_degradation

    i, j = divmod(np.argmax(degradation_matrix), cluster_count)

    cluster_masks.append((cluster_masks[i] | cluster_masks[j]))
    cluster_lkhd.append(degradation_matrix[i, j] + cluster_lkhd[i] + cluster_lkhd[j])

    del cluster_masks[j]
    del cluster_masks[i]
    del cluster_lkhd[j]
    del cluster_lkhd[i]

    cluster_count -= 1


grouped_labels = np.empty_like(labels)
for i, mask in enumerate(cluster_masks):
    grouped_labels[mask] = i


m = defaultdict(set)
for l, g in zip(labels, grouped_labels):
    m[g].add(l)

