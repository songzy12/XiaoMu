

def compute_mrr(scores, match):
    mrr = 0
    for k, v in match.items():
        rank_list = scores[k]
        rank = min([rank_list.index(x) if x in rank_list else -1 for x in v])
        mrr += 1. / (rank+1) if rank != -1 else 0
    return mrr * 1. / len(match)


def compute_hit_at_n(scores, match, n):
    cnt = 0
    for k, v in match.items():
        rank_list = scores[k][:n]
        if set(rank_list) & set(v):
            cnt += 1
    return cnt * 1. / len(match)
