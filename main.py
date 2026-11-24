"""Small offline water-consumption anomaly detection demo."""
from pathlib import Path
import csv
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, confusion_matrix

ROOT = Path(__file__).resolve().parent


def simulate(days, seed):
    rng = np.random.default_rng(seed)
    hours = np.arange(days * 24) % 24
    # Approximate hourly totals in litres, not real household data.
    profile = np.array([0.1, 0.1, 0.1, 0.1, 0.2, 1, 12, 28, 20, 8, 6, 7,
                        12, 8, 6, 6, 8, 12, 22, 25, 16, 8, 3, 0.5])
    water = rng.gamma(shape=2, scale=profile[hours] / 2)
    return hours, water


def features(hours, water, typical=None):
    # Encode time cyclically so midnight is not a discontinuity.
    phase = 2 * np.pi * hours / 24
    if typical is None:
        typical = np.ones(24)
    # Compare consumption against the expected amount at this hour.
    relative = np.log1p(water / (typical[hours] + 0.2))
    previous = np.r_[0., relative[:-1]]
    rolling = np.convolve(np.r_[0., 0., relative], np.ones(3)/3, mode='valid')
    return np.column_stack([relative, previous, rolling, np.sin(phase), np.cos(phase)])


def evaluate():
    h_train, w_train = simulate(60, 42)
    h_test, w_test = simulate(14, 43)
    truth = np.zeros(len(w_test), dtype=int)
    # Two overnight continuous-flow events and one daytime spike.
    incidents = [(3*24, 3*24+5, 17), (7*24+13, 7*24+14, 180),
                 (10*24, 10*24+5, 12)]
    for start, end, extra in incidents:
        w_test[start:end] += extra
        truth[start:end] = 1

    model = IsolationForest(n_estimators=200, contamination=0.025, random_state=42)
    typical = np.array([np.median(w_train[h_train == h]) for h in range(24)])
    model.fit(features(h_train, w_train, typical)[2:])
    x_test = features(h_test, w_test, typical)
    scores = -model.decision_function(x_test)
    predicted = (model.predict(x_test) == -1).astype(int)
    return h_test, w_test, truth, predicted, scores


def main():
    hours, water, truth, predicted, scores = evaluate()
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
    print('SmartHome AI Guardian - synthetic test')
    print(f'Test hours: {len(water)} | injected anomaly hours: {truth.sum()}')
    print(f'Precision: {precision_score(truth, predicted, zero_division=0):.3f}')
    print(f'Recall: {recall_score(truth, predicted, zero_division=0):.3f}')
    print(f'TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}')
    output = ROOT / 'results.csv'
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['day', 'hour', 'litres', 'injected_anomaly', 'predicted_anomaly', 'anomaly_score'])
        for i, (h, w, actual, flag, score) in enumerate(zip(hours, water, truth, predicted, scores)):
            writer.writerow([i//24+1, int(h), round(float(w), 3), int(actual), int(flag), round(float(score), 6)])
    print(f'Results saved to {output.name}')


if __name__ == '__main__':
    main()
