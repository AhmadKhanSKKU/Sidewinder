import pickle
with open("/media/ahmad/Work/Experiments/Pandora/NetMasquerade-main/src/pandora_ddos_ackFrag_experiment.pkl", "rb") as f:
    data = pickle.load(f)

max_ipd = 0
max_size = 0
for flow in data:
    max_ipd = max(max_ipd, max(flow[0]))
    max_size = max(max_size, max(flow[1]))

print(f"Minimum required timevocab_size: {max_ipd + 1}")
print(f"Minimum required sizevocab_size: {max_size + 1}")