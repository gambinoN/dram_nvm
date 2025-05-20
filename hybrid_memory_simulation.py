"""
Hybrid Memory System Simulation (DRAM + NVM)
This simulation models a memory system that combines DRAM and Non-Volatile Memory (NVM),
focusing on performance characteristics, wear-leveling, and persistence tradeoffs.
Configurable via command-line arguments or a JSON config file.
"""

import argparse
import json
import os
import time
import random
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class MemoryCell:
    """Base class for memory cells"""
    def __init__(self, address):
        self.address = address
        self.data = 0
        self.last_access_time = 0


class DRAMCell(MemoryCell):
    """DRAM cell with volatility and fast access"""
    def __init__(self, address):
        super().__init__(address)
        self.refresh_time = 0  # Last refresh timestamp

    def needs_refresh(self, current_time, refresh_interval=64):
        """Check if this cell needs a refresh"""
        return (current_time - self.refresh_time) > refresh_interval

    def refresh(self, current_time):
        """Refresh the DRAM cell"""
        self.refresh_time = current_time


class NVMCell(MemoryCell):
    """NVM cell with persistence and limited endurance"""
    def __init__(self, address, max_writes=10000):
        super().__init__(address)
        self.write_count = 0
        self.max_writes = max_writes

    def write(self, data):
        """Write data and increment wear counter if data changes"""
        if self.data != data:
            self.data = data
            self.write_count += 1
            return True
        return False

    def get_wear_level(self):
        """Return wear as percentage of max writes"""
        return (self.write_count / self.max_writes) * 100

    def is_worn_out(self):
        """Check if this cell has exceeded endurance"""
        return self.write_count >= self.max_writes


class MemoryController:
    """Controller for hybrid DRAM+NVM memory"""
    def __init__(self, dram_size_mb, nvm_size_mb, page_size_kb=4):
        # Convert sizes
        self.dram_size = dram_size_mb * 1024 * 1024
        self.nvm_size = nvm_size_mb * 1024 * 1024
        self.page_size = page_size_kb * 1024

        # Pages count
        self.dram_page_count = self.dram_size // self.page_size
        self.nvm_page_count = self.nvm_size // self.page_size

        # Device cells
        self.dram = {i: DRAMCell(i) for i in range(self.dram_page_count)}
        self.nvm  = {i: NVMCell(i) for i in range(self.nvm_page_count)}

        # Logical→physical maps
        self.address_to_dram = {}
        self.address_to_nvm  = {}

        # Stats
        self.stats = {
            'dram_reads': 0,
            'dram_writes': 0,
            'nvm_reads': 0,
            'nvm_writes': 0,
            'dram_misses': 0,
            'page_migrations': 0,
            'nvm_wear_levels': defaultdict(int)
        }

        # Simulation clock
        self.current_time = 0

        # Access freq for wear-leveling & migrations
        self.access_frequency = defaultdict(int)

    def allocate(self, size_bytes, prefer_nvm=False):
        """Allocate pages for a logical region"""
        pages_needed = (size_bytes + self.page_size - 1) // self.page_size
        if prefer_nvm and len(self.address_to_nvm) + pages_needed <= self.nvm_page_count:
            start = len(self.address_to_nvm)
            for i in range(pages_needed):
                self.address_to_nvm[start + i] = i
            return start
        else:
            start = len(self.address_to_dram)
            for i in range(pages_needed):
                self.address_to_dram[start + i] = i
            return start

    def read(self, address):
        """Read from logical address"""
        self.current_time += 1
        self.access_frequency[address] += 1

        # DRAM hit
        if address in self.address_to_dram:
            pa = self.address_to_dram[address]
            cell = self.dram[pa]
            cell.last_access_time = self.current_time
            self.stats['dram_reads'] += 1
            return cell.data

        # NVM hit
        if address in self.address_to_nvm:
            pa = self.address_to_nvm[address]
            cell = self.nvm[pa]
            cell.last_access_time = self.current_time
            self.stats['nvm_reads'] += 1
            # Migrate hot pages
            if self.access_frequency[address] > 10:
                self.migrate_to_dram(address)
            return cell.data

        # Miss
        self.stats['dram_misses'] += 1
        return None

    def write(self, address, data):
        """Write to logical address"""
        self.current_time += 1
        self.access_frequency[address] += 1

        if address in self.address_to_dram:
            pa = self.address_to_dram[address]
            cell = self.dram[pa]
            cell.data = data
            cell.last_access_time = self.current_time
            self.stats['dram_writes'] += 1
            return True

        if address in self.address_to_nvm:
            pa = self.address_to_nvm[address]
            cell = self.nvm[pa]
            if cell.write(data):
                self.stats['nvm_writes'] += 1
                self.stats['nvm_wear_levels'][address] = cell.get_wear_level()
                if cell.get_wear_level() > 70:
                    self.wear_leveling(address)
            cell.last_access_time = self.current_time
            return True

        return False

    def migrate_to_dram(self, address):
        """Move a page from NVM to DRAM"""
        if address not in self.address_to_nvm:
            return False
        npa = self.address_to_nvm[address]
        data = self.nvm[npa].data
        # Find DRAM slot
        dpa = len(self.address_to_dram)
        if dpa >= self.dram_page_count:
            evict = self._find_lru_dram()
            dpa = self.address_to_dram[evict]
            del self.address_to_dram[evict]
        self.dram[dpa].data = data
        self.address_to_dram[address] = dpa
        del self.address_to_nvm[address]
        self.stats['page_migrations'] += 1
        return True

    def _find_lru_dram(self):
        """Least recently used logical addr in DRAM"""
        lru, min_t = None, float('inf')
        for addr, pa in self.address_to_dram.items():
            if self.dram[pa].last_access_time < min_t:
                lru = addr
                min_t = self.dram[pa].last_access_time
        return lru

    def wear_leveling(self, address):
        """Swap a worn NVM cell with a less-worn target"""
        if address not in self.address_to_nvm:
            return False
        src_pa = self.address_to_nvm[address]
        src_cell = self.nvm[src_pa]
        # find target
        tgt_pa, min_w = None, src_cell.write_count
        for pa, cell in self.nvm.items():
            if cell.write_count < min_w and pa != src_pa:
                tgt_pa, min_w = pa, cell.write_count
        if tgt_pa is None:
            return False
        # find logical mapping for tgt
        tgt_la = next(la for la, pa in self.address_to_nvm.items() if pa == tgt_pa)
        # swap
        self.address_to_nvm[address], self.address_to_nvm[tgt_la] = tgt_pa, src_pa
        data_src = src_cell.data
        src_cell.data, self.nvm[tgt_pa].data = self.nvm[tgt_pa].data, data_src
        return True

    def refresh_dram(self):
        """Refresh DRAM cells as needed"""
        cnt = 0
        for pa, cell in self.dram.items():
            if cell.needs_refresh(self.current_time):
                cell.refresh(self.current_time)
                cnt += 1
        return cnt

    def get_memory_state(self):
        """Snapshot of key metrics"""
        wear = self.stats['nvm_wear_levels']
        return {
            'dram_usage': len(self.address_to_dram)/self.dram_page_count*100,
            'nvm_usage': len(self.address_to_nvm)/self.nvm_page_count*100,
            'nvm_wear_avg': sum(wear.values())/max(1,len(wear)),
            'nvm_wear_max': max(wear.values()) if wear else 0,
            'dram_reads': self.stats['dram_reads'],
            'dram_writes': self.stats['dram_writes'],
            'nvm_reads': self.stats['nvm_reads'],
            'nvm_writes': self.stats['nvm_writes'],
            'page_migrations': self.stats['page_migrations'],
        }


class WorkloadGenerator:
    """Generates memory workloads"""
    def __init__(self, address_space_size, seed=None):
        self.address_space_size = address_space_size
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate_random_workload(self, num_ops, write_ratio=0.3):
        return [(
            'write' if random.random()<write_ratio else 'read',
            random.randint(0,self.address_space_size-1),
            random.randint(0,255) if random.random()<write_ratio else None
        ) for _ in range(num_ops)]

    def generate_sequential_workload(self, num_ops, write_ratio=0.3):
        return [(
            'write' if random.random()<write_ratio else 'read',
            i % self.address_space_size,
            random.randint(0,255) if random.random()<write_ratio else None
        ) for i in range(num_ops)]

    def generate_zipfian_workload(self, num_ops, alpha=1.0, write_ratio=0.3):
        x = np.arange(1,self.address_space_size+1)
        w = x**(-alpha)
        w /= w.sum()
        addrs = np.random.choice(np.arange(self.address_space_size), size=num_ops, p=w)
        return [(
            'write' if random.random()<write_ratio else 'read',
            addr,
            random.randint(0,255) if random.random()<write_ratio else None
        ) for addr in addrs]

    def generate_loop_workload(self, num_ops, loop_size=100, write_ratio=0.3):
        return [(
            'write' if random.random()<write_ratio else 'read',
            i%loop_size,
            random.randint(0,255) if random.random()<write_ratio else None
        ) for i in range(num_ops)]


class HybridMemorySimulation:
    """Runs workloads and aggregates results"""
    def __init__(self, dram_size_mb=256, nvm_size_mb=1024, page_size_kb=4):
        # Initialize controller
        self.controller = MemoryController(dram_size_mb, nvm_size_mb, page_size_kb)
        # Pre-allocate entire memory space: DRAM then NVM
        # Map DRAM pages
        self.controller.allocate(self.controller.dram_size, prefer_nvm=False)
        # Map NVM pages
        self.controller.allocate(self.controller.nvm_size, prefer_nvm=True)
        # Address space size = total pages
        total_pages = self.controller.dram_page_count + self.controller.nvm_page_count
        self.gen = WorkloadGenerator(total_pages, seed=None)
        self.results = []

    def run_workload(self, workload, description=""):
        start = time.time()
        init = self.controller.get_memory_state()
        for op, addr, data in workload:
            if op=='read': self.controller.read(addr)
            else: self.controller.write(addr,data)
            if self.controller.current_time%100==0:
                self.controller.refresh_dram()
        final = self.controller.get_memory_state()
        elapsed = time.time()-start
        stats = self.controller.stats
        res = {
            'description': description,
            'workload_size': len(workload),
            'elapsed_time': elapsed,
            'ops_per_sec': len(workload)/elapsed,
            **init, **final,
            'dram_miss_rate': stats['dram_misses']/len(workload),
            'nvm_wear_levels': dict(stats['nvm_wear_levels'])
        }
        self.results.append(res)
        return res

    def reset_stats(self):
        self.controller.stats = {
            'dram_reads':0,'dram_writes':0,'nvm_reads':0,'nvm_writes':0,
            'dram_misses':0,'page_migrations':0,'nvm_wear_levels':defaultdict(int)
        }

    def compare_workloads(self, wdict):
        outs=[]
        for name,w in wdict.items():
            self.reset_stats()
            outs.append(self.run_workload(w,name))
        return outs

    def visualize_results(self, results=None, save_dir='results'):
        if results is None: results=self.results
        os.makedirs(save_dir,exist_ok=True)
        ts=datetime.now().strftime('%Y%m%d_%H%M%S')
        df=pd.DataFrame(results)
        # ... plots as before
        # Save summary CSV
        df.to_csv(f"{save_dir}/summary_{ts}.csv", index=False)
        print(f"Results in {save_dir}")
        return df


# ---- CLI & Config ----

def parse_args():
    p=argparse.ArgumentParser("Hybrid DRAM+NVM Simulator")
    p.add_argument('--config',type=str,help='JSON config path')
    p.add_argument('--dram-size-mb',type=int,help='DRAM size in MiB')
    p.add_argument('--nvm-size-mb',type=int,help='NVM size in MiB')
    p.add_argument('--page-size-kb',type=int,default=4,help='Page size KiB')
    p.add_argument('--num-ops',type=int,default=100000,help='Ops per workload')
    p.add_argument('--write-ratio',type=float,default=0.3,help='Write fraction')
    p.add_argument('--loop-size',type=int,default=100,help='Loop workload size')
    p.add_argument('--seed',type=int,help='Random seed')
    p.add_argument('--output-dir',type=str,default='results',help='Output directory')
    p.add_argument('--workloads',nargs='+',choices=['random','sequential','zipfian','loop'],
                   default=['random','sequential','zipfian','loop'],help='Which workloads')
    return p.parse_args()


def load_config(path):
    with open(path) as f: return json.load(f)


def merge_config(args,cfg):
    for k,v in cfg.items():
        a=k.replace('-','_')
        if hasattr(args,a) and getattr(args,a) is None:
            setattr(args,a,v)
    return args


def main():
    args=parse_args()
    if args.config:
        cfg=load_config(args.config)
        args=merge_config(args,cfg)
    if args.seed is not None:
        random.seed(args.seed); np.random.seed(args.seed)
    sim=HybridMemorySimulation(
        dram_size_mb=args.dram_size_mb or 256,
        nvm_size_mb=args.nvm_size_mb or 1024,
        page_size_kb=args.page_size_kb
    )
    workloads={}
    n=args.num_ops; r=args.write_ratio; ls=args.loop_size
    if 'random' in args.workloads:
        workloads['Random']=sim.gen.generate_random_workload(n,write_ratio=r)
    if 'sequential' in args.workloads:
        workloads['Sequential']=sim.gen.generate_sequential_workload(n,write_ratio=r)
    if 'zipfian' in args.workloads:
        workloads['Zipfian']=sim.gen.generate_zipfian_workload(n,write_ratio=r)
    if 'loop' in args.workloads:
        workloads['Loop']=sim.gen.generate_loop_workload(n,loop_size=ls,write_ratio=r)
    results=sim.compare_workloads(workloads)
    sim.visualize_results(results,save_dir=args.output_dir)

if __name__=='__main__':
    main()