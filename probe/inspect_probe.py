import pickle
with open("probe/detector.pt", "rb") as f:
    probe = pickle.load(f)

print(type(probe))
print(probe.keys() if hasattr(probe, "keys") else "not a dict")
for k, v in probe.items():
    try:
        print(f"{k:20s} {type(v).__name__:10s} shape={tuple(v.shape)} dtype={v.dtype}")
    except AttributeError:
        print(f"{k:20s} {type(v).__name__:10s} value={v}")