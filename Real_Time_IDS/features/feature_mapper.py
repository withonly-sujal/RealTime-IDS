FEATURE_ORDER = [
    'dur',
    'proto',
    'state',
    'spkts',
    'dpkts',
    'sbytes',
    'dbytes',
    'rate',
    'sttl',
    'dttl',
    'sload',
    'dload',
    'sinpkt',
    'dinpkt',
    'sjit',
    'djit',
    'smean',
    'dmean',
    'ct_srv_src',
    'ct_state_ttl',
    'ct_dst_ltm',
    'ct_src_dport_ltm',
    'ct_dst_sport_ltm',
    'ct_dst_src_ltm',
    'ct_src_ltm',
    'ct_srv_dst'
]


def prepare_features(df):
    for col in FEATURE_ORDER:
        if col not in df:
            df[col] = 0

    return df[FEATURE_ORDER]