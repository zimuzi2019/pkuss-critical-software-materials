#!/usr/bin/python
#

from __future__ import print_function
from bcc import BPF
from bcc.utils import printb

# load BPF program
b = BPF(text="""
#include <uapi/linux/ptrace.h>

BPF_HASH(last);

int do_trace(struct pt_regs *ctx) {
    u64 ts, *tsp, delta, key = 0;

    // attempt to read stored timestamp
    tsp = last.lookup(&key);
    if (tsp != NULL) {
        ts = *tsp + 1 ;
    }
    else {
        ts = 1;
    }
    last.update(&key, &ts);
    bpf_trace_printk("%d\\n", ts);
    return 0;
}
""")

b.attach_kprobe(event=b.get_syscall_fnname("socket"), fn_name="do_trace")
print("Tracing for socket... Ctrl-C to end")

# format output
start = 0
while 1:
    try:
        (task, pid, cpu, flags, ts, count) = b.trace_fields()
        printb(b"At time %.2f s: socket detected, count %s" % (ts, count))
    except KeyboardInterrupt:
        exit()