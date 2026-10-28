"""Episode-level premature completion metrics for a tactile-only classifier."""
import numpy as np


def trigger_metrics(uid, labels, done_probability, threshold, consecutive=1):
    uid, labels, probability = map(np.asarray, (uid, labels, done_probability))
    if not (uid.shape == labels.shape == probability.shape):
        raise ValueError('Inputs must be equal-length one-dimensional arrays')
    if consecutive < 1:
        raise ValueError("consecutive must be positive")
    total = premature = correct = missed = positive = 0
    order=np.argsort(uid,kind="stable")
    ordered_uid=uid[order]
    boundaries=np.r_[0,np.flatnonzero(np.diff(ordered_uid))+1,len(uid)]
    for begin,end in zip(boundaries[:-1],boundaries[1:]):
        if begin==end:continue
        indices=order[begin:end]
        truth,pred=labels[indices],probability[indices]
        flags=pred>=threshold
        cumulative=np.r_[0,np.cumsum(flags,dtype=np.int64)]
        windows=cumulative[consecutive:]-cumulative[:-consecutive] if len(pred)>=consecutive else np.empty(0)
        triggers=np.flatnonzero(windows==consecutive)[:1]+consecutive-1
        has_done = (truth == 1).any()
        positive += int(has_done)
        total += 1
        if len(triggers):
            if truth[triggers[0]] == 1:
                correct += 1
            else:
                premature += 1
                missed += int(has_done)
        else:
            missed += int(has_done)
    return {'threshold': threshold, 'consecutive': consecutive, 'episodes': total, 'positive_episodes': positive,
            'premature_triggers': premature, 'correct_first_triggers': correct,
            'missed_positive_episodes': missed,
            'premature_episode_rate': premature/max(total, 1),
            'positive_episode_recall': correct/positive if positive else None}
