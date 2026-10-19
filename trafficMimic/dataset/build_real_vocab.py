# NetMasquerade-main/src/trafficMimic/dataset/build_real_vocab.py

import os
import sys
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from scapy.all import PcapReader
from dataset.vocab import TimeVocab, SizeVocab
from typing import List, Tuple, Dict, Any

def process_pcap_to_flows(
    pcap_path: str, max_flows: int = 20000
) -> Tuple[List[List[float]], List[List[int]]]:
    print(f"[*] Reading PCAP: {pcap_path}")
    # Allow flexible flow keys (IP tuples or ARP/L2 tuples)
    flows: Dict[Any, List[Tuple[float, int]]] = {}

    pkt_count = 0
    with PcapReader(pcap_path) as pcap_reader:
        for pkt in pcap_reader:
            pkt_count += 1
            if pkt_count % 100000 == 0:
                print(
                    f"    -> Processed {pkt_count} packets... tracked flows: {len(flows)}"
                )
            if pkt_count > 500000:
                break

            # 1. IP Traffic (TCP/UDP)
            if pkt.haslayer("IP"):
                ip_layer = pkt["IP"]
                src_ip, dst_ip, proto = ip_layer.src, ip_layer.dst, ip_layer.proto
                src_port, dst_port = 0, 0

                if proto == 6 and pkt.haslayer("TCP"):  # 6 is TCP
                    src_port, dst_port = pkt["TCP"].sport, pkt["TCP"].dport
                elif proto == 17 and pkt.haslayer("UDP"):  # 17 is UDP
                    src_port, dst_port = pkt["UDP"].sport, pkt["UDP"].dport

                flow_key = (src_ip, dst_ip, src_port, dst_port, proto)

            # 2. ARP Traffic (Grouped by Hardware Src & Target IP)
            elif pkt.haslayer("ARP"):
                arp_layer = pkt["ARP"]
                flow_key = (arp_layer.hwsrc, arp_layer.pdst, "ARP")

            # 3. Layer 2 Fallback (Grouped by MAC addresses)
            elif pkt.haslayer("Ether"):
                eth_layer = pkt["Ether"]
                flow_key = (eth_layer.src, eth_layer.dst, "L2")

            else:
                continue

            if flow_key not in flows:
                flows[flow_key] = []

            flows[flow_key].append((float(pkt.time), len(pkt)))

    print(f"[+] Total packets read: {pkt_count}")
    print(f"[+] Found {len(flows)} unique raw streams.")

    all_ipd_sequences: List[List[float]] = []
    all_size_sequences: List[List[int]] = []

    MAX_PACKETS_PER_FLOW = 100
    # STRIDE determines overlap. A stride of 10 shifts the window by 10 packets.
    # For a 35,000 packet stream, a stride of 10 generates ~3,500 flows.
    STRIDE = 1

    # Dynamic sorting and slicing loop
    for _, packets in flows.items():
        if len(all_ipd_sequences) >= max_flows:
            break

        # Filter out short noise flows entirely
        if len(packets) < 3:
            continue

        packets.sort(key=lambda x: x[0])

        # Use a sliding window to generate overlapping sequences from long streams
        for chunk_start in range(0, len(packets), STRIDE):
            chunk = packets[chunk_start : chunk_start + MAX_PACKETS_PER_FLOW]

            # Ensure the chunk itself meets the minimum length requirement
            if len(chunk) < 3:
                continue

            ipds = [0.0]
            sizes = [chunk[0][1]]

            for i in range(1, len(chunk)):
                ipd = max(1e-6, chunk[i][0] - chunk[i - 1][0])
                ipds.append(ipd)
                sizes.append(chunk[i][1])

            all_ipd_sequences.append(ipds)
            all_size_sequences.append(sizes)

            # Break early if we hit the global max_flows limit while chunking
            if len(all_ipd_sequences) >= max_flows:
                break

    print(f"[+] Kept {len(all_ipd_sequences)} valid flows for training.")
    return all_ipd_sequences, all_size_sequences


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract flows from PCAP")
    parser.add_argument("--pcap", type=str, required=True, help="Input PCAP file")
    parser.add_argument("--out", type=str, required=True, help="Output PKL file")
    args = parser.parse_args()

    ipd_data, size_data = process_pcap_to_flows(args.pcap)

    if len(ipd_data) == 0:
        print(f"[!] ERROR: 0 flows survived the filter for {args.pcap}.")
        sys.exit(1)

    with open(args.out, "wb") as f:
        combined_data = list(zip(ipd_data, size_data))
        pickle.dump(combined_data, f)
    print(f"[+] Successfully saved {len(combined_data)} flows to {args.out}")

    # print("[*] Generating TimeVocab...")
    # time_vocab = TimeVocab(ipd_data, max_size=56)
    # time_vocab.save_vocab("timevocab_56.pkl")

    # print("[*] Generating SizeVocab...")
    # size_vocab = SizeVocab(size_data)
    # size_vocab.save_vocab("sizevocab_1606.pkl")
