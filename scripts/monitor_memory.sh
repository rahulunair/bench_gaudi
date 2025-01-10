#!/bin/bash

# Monitor memory usage of Gaudi devices
watch -n 1 hl-smi --query-aip=memory.total,memory.used,memory.free --format=csv
