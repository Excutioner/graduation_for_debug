def load_cmcs(cmc_file):
    """Load CMCS data from file."""
    cmcs_packets = []
    with open(cmc_file, 'r') as f:
        for line in f.readlines():
            line = line.split()
            # 将line转换为float
            line[1:] = [float(x) for x in line[1:]]
            line[:1] = [int(float(x)) for x in line[:1]]
            cmcs_packets.append(line)
    return cmcs_packets