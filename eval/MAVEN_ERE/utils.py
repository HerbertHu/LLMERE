import copy


def get_clusters(event2cluster):
    # set remove duplication
    clusters = list(set(event2cluster.values()))
    return clusters


def get_id2clusters(doc_pairs, doc_pred, doc_events):
    """
    obtain coreference clusters
    """

    idx_to_clusters = {}
    for event_id in doc_events:
        item_set = set()
        item_set.add(event_id)
        idx_to_clusters[event_id] = item_set

    coref_pairs = set()
    for pair_item, pair_pred in zip(doc_pairs, doc_pred):
        if pair_pred == 1:
            coref_pairs.add((pair_item[0], pair_item[1]))

    new_coref_pairs = copy.deepcopy(coref_pairs)
    for item in coref_pairs:
        reverse_item = (item[1], item[0])
        if reverse_item not in coref_pairs:
            new_coref_pairs.remove(item)

    # remove duplicate elements
    samplify_coref_pairs = set()
    for item in new_coref_pairs:
        if item[0] < item[1]:
            samplify_coref_pairs.add(item)
        else:
            reverse_item = (item[1], item[0])
            samplify_coref_pairs.add(reverse_item)

    # set union
    for item in samplify_coref_pairs:
        union_cluster = set.union(idx_to_clusters[item[0]], idx_to_clusters[item[1]])
        for j in union_cluster:
            idx_to_clusters[j] = union_cluster

    idx_to_clusters = {i: tuple(sorted(idx_to_clusters[i])) for i in idx_to_clusters}
    predicted_clusters = get_clusters(idx_to_clusters)
    return predicted_clusters, idx_to_clusters


