"""
Efficient on-demand generation of subsets with top-k largest sums.

Inspired by paper [[Efficient Reporting of Top-k Subset Sums]] (Sanyal et al., 2021).
Although they claim their algorithm works for any set of real numbers, presence of
negative numbers render the algorithm inapplicable. This is a modification of their
algorithm that can handle sets with real numbers by keeping a 'dual' metadata
structure that contains two instances of the metadata structure outlined in the
original paper.
"""
import heapq


def topk_subset_gen(values):
    """
    Keep generating subsets of given list where each element is a real number, so
    that subsets with largest value sums are returned from the top. Return found
    subsets as index lists, along with their sum values.
    """
    # Divide into non-negative vs. negative parts, then sort (We'll refer to the
    # non-negative part as "pos" ... if that could be forgiven :] )
    values_sorted_pos = filter(lambda x: x[1] >= 0, enumerate(values))
    values_sorted_neg = filter(lambda x: x[1] < 0, enumerate(values))

    values_sorted_pos = sorted(values_sorted_pos, key=lambda x: x[1])
    values_sorted_neg = sorted(values_sorted_neg, key=lambda x: x[1], reverse=True)

    # The dual metadata structure, which is basically a DAG where edges are implied by
    # node values, doesn't need to be explicitly constructed and stored, as we can
    # deductively determine nodes and associated data on the fly. Just specify the root.
    bit_pattern = ((0,)*len(values_sorted_pos), (0,)*len(values_sorted_neg))
    G_root_data = {
        "p_pos": [-1, -1, -1],
        "p_neg": [-1, -1, -1],
        "size_pos": 0,
        "size_neg": 0,
        "sum_value": \
            sum([x[1] for i, x in enumerate(values_sorted_pos) if not bit_pattern[0][i]]) + \
            sum([x[1] for i, x in enumerate(values_sorted_neg) if bit_pattern[1][i]])
    }

    # Max-heap for comparisons of values not directly connected by a path in G
    # (Values must be sign-flipped as heapq only provides min-heap!)
    H = []
    heapq.heappush(H, (-G_root_data["sum_value"], bit_pattern, G_root_data))

    # Loop until H (G) is exhausted
    visited = set()
    while len(H) > 0:
        # Extract maximum 
        current_sum, current_node, current_data = heapq.heappop(H)

        if current_node not in visited:
            visited.add(current_node)

        to_push = _expand_node(
            current_node, current_data,
            [x[1] for x in values_sorted_pos], [x[1] for x in values_sorted_neg]
        )

        # Push expanded nodes to G, H
        for bits, data in to_push:
            if bits not in visited:
                heapq.heappush(H, (-data["sum_value"], bits, data))

        current_subset = \
            [values_sorted_pos[i][0] for i, b in enumerate(current_node[0]) if not b] + \
            [values_sorted_neg[i][0] for i, b in enumerate(current_node[1]) if b]
        current_subset = set(current_subset)

        yield (current_subset, -current_sum)


def _expand_node(bit_pattern, data, vals_pos, vals_neg):
    """ Return set of bit patterns to add to frontier (i.e. heap) """
    nodes = []
    bits_pos, bits_neg = bit_pattern

    ## Positive side
    if len(vals_pos) > 0:
        p1, p2, p3 = data["p_pos"]

        # Static one-shift; case 1
        if 0 < p1 < len(bits_pos)-1 and bits_pos[p1+1] == 0:
            new_bits_pos = bits_pos[:p1] + (0,1) + bits_pos[p1+2:]
            new_data = {
                "p_pos": [p1+1, p2, max(p1+1, p3)],
                "p_neg": data["p_neg"],
                "size_pos": data["size_pos"],
                "size_neg": data["size_neg"],
                "sum_value": data["sum_value"] + vals_pos[p1] - vals_pos[p1+1]
            }
            nodes.append(((new_bits_pos, bits_neg), new_data))

        # Static one-shift; case 2
        if -1 < p2 < len(bits_pos)-1 and bits_pos[p2+1] == 0:
            new_bits_pos = bits_pos[:p2] + (0,1) + bits_pos[p2+2:]
            new_data = {
                "p_pos": [p2+1, p2-1, max(p2+1, p3)],
                "p_neg": data["p_neg"],
                "size_pos": data["size_pos"],
                "size_neg": data["size_neg"],
                "sum_value": data["sum_value"] + vals_pos[p2] - vals_pos[p2+1]
            }
            nodes.append(((new_bits_pos, bits_neg), new_data))

        # Incremental shift; case 1 (root only)
        if p1 == -1 and p2 == -1 and p3 == -1:
            new_bits_pos = (1,) + ((0,) * (len(bits_pos)-1))
            new_data = {
                "p_pos": [-1, 0, 0],
                "p_neg": data["p_neg"],
                "size_pos": data["size_pos"] + 1,
                "size_neg": data["size_neg"],
                "sum_value": data["sum_value"] - vals_pos[0]
            }
            nodes.append(((new_bits_pos, bits_neg), new_data))

        # Incremental shift; case 2
        if p1 == 1 and p2 == -1 and p3 == data["size_pos"]:
            new_bits_pos = (1,) + bits_pos[p1:]
            new_data = {
                "p_pos": [-1, p3, p3],
                "p_neg": data["p_neg"],
                "size_pos": data["size_pos"] + 1,
                "size_neg": data["size_neg"],
                "sum_value": data["sum_value"] - vals_pos[0]
            }
            nodes.append(((new_bits_pos, bits_neg), new_data))

    ## Negative side
    if len(vals_neg) > 0:
        p1, p2, p3 = data["p_neg"]

        # Static one-shift; case 1
        if 0 < p1 < len(bits_neg)-1 and bits_neg[p1+1] == 0:
            new_bits_neg = bits_neg[:p1] + (0,1) + bits_neg[p1+2:]
            new_data = {
                "p_pos": data["p_pos"],
                "p_neg": [p1+1, 0, max(p1+1, p3)],
                "size_pos": data["size_pos"],
                "size_neg": data["size_neg"],
                "sum_value": data["sum_value"] - vals_neg[p1] + vals_neg[p1+1]
            }
            nodes.append(((bits_pos, new_bits_neg), new_data))

        # Static one-shift; case 2
        if -1 < p2 < len(bits_neg)-1 and bits_neg[p2+1] == 0:
            new_bits_neg = bits_neg[:p2] + (0,1) + bits_neg[p2+2:]
            new_data = {
                "p_pos": data["p_pos"],
                "p_neg": [p2+1, p2-1, max(p2+1, p3)],
                "size_pos": data["size_pos"],
                "size_neg": data["size_neg"],
                "sum_value": data["sum_value"] - vals_neg[p2] + vals_neg[p2+1]
            }
            nodes.append(((bits_pos, new_bits_neg), new_data))
        
        # Incremental shift; case 1 (root only)
        if p1 == -1 and p2 == -1 and p3 == -1:
            new_bits_neg = (1,) + ((0,) * (len(bits_neg)-1))
            new_data = {
                "p_pos": data["p_pos"],
                "p_neg": [-1, 0, 0],
                "size_pos": data["size_pos"],
                "size_neg": data["size_neg"] + 1,
                "sum_value": data["sum_value"] + vals_neg[0]
            }
            nodes.append(((bits_pos, new_bits_neg), new_data))

        # Incremental shift; case 2
        if p1 == 1 and p2 == -1 and p3 == data["size_neg"]:
            new_bits_neg = (1,) + bits_neg[p1:]
            new_data = {
                "p_pos": data["p_pos"],
                "p_neg": [-1, p3, p3],
                "size_pos": data["size_pos"],
                "size_neg": data["size_neg"] + 1,
                "sum_value": data["sum_value"] + vals_neg[0]
            }
            nodes.append(((bits_pos, new_bits_neg), new_data))

    return nodes
