from utils.helpers import ERROR, LOG

def get_block_config(num_layers):
    assert num_layers >= 2, f"{ERROR}Number of layers must be at least 2 for a minimal ResNet configuration"

    if num_layers == 18:
        return [2, 2, 2, 2]
    elif num_layers == 34:
        return [3, 4, 6, 3]
    elif num_layers == 50:
        return [3, 4, 6, 3]
    elif num_layers == 101:
        return [3, 4, 23, 3]
    elif num_layers == 152:
        return [3, 8, 36, 3]
    elif num_layers == 1202:
        return [200, 200, 200, 200]

    if (num_layers - 2) % 6 != 0:
        raise ValueError(f"{ERROR}Invalid number of layers for ResNet. It should be 2 plus 6n (n >= 1)")

    n = (num_layers - 2) // 6
    block_config = [n] * 4
    print(f"{LOG}Block configuration for ResNet{num_layers}: {block_config}")
    return block_config