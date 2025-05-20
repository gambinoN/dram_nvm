"""
Hybrid Memory System Simulation (DRAM + NVM)
This simulation models a memory system that combines DRAM and Non-Volatile Memory (NVM),
focusing on performance characteristics, wear-leveling, and persistence tradeoffs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
from collections import defaultdict
import os
from datetime import datetime

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
        self.refresh_time = 0  # Time when this cell was last refreshed
        
    def needs_refresh(self, current_time, refresh_interval=64):
        """Check if the cell needs to be refreshed"""
        return current_time - self.refresh_time > refresh_interval
        
    def refresh(self, current_time):
        """Refresh the DRAM cell"""
        self.refresh_time = current_time
        # In a real system, this would re-read and rewrite the data

class NVMCell(MemoryCell):
    """NVM cell with persistence and limited write endurance"""
    def __init__(self, address, max_writes=10000):
        super().__init__(address)
        self.write_count = 0
        self.max_writes = max_writes  # Maximum number of writes before failure
        
    def write(self, data):
        """Write data to the NVM cell and increment write counter"""
        # Only count as a write if the data actually changes
        if self.data != data:
            self.data = data
            self.write_count += 1
            return True
        return False
            
    def get_wear_level(self):
        """Return the current wear level as a percentage"""
        return (self.write_count / self.max_writes) * 100
    
    def is_worn_out(self):
        """Check if the cell has exceeded its write endurance"""
        return self.write_count >= self.max_writes

class MemoryController:
    """Controller for hybrid memory system"""
    def __init__(self, dram_size_mb, nvm_size_mb, page_size_kb=4):
        # Convert sizes to bytes for internal calculations
        self.dram_size = dram_size_mb * 1024 * 1024
        self.nvm_size = nvm_size_mb * 1024 * 1024
        self.page_size = page_size_kb * 1024
        
        # Calculate number of pages
        self.dram_page_count = self.dram_size // self.page_size
        self.nvm_page_count = self.nvm_size // self.page_size
        
        # Initialize DRAM and NVM
        self.dram = {}
        self.nvm = {}
        
        # Allocation maps
        self.address_to_dram = {}  # Maps logical addresses to DRAM cells
        self.address_to_nvm = {}   # Maps logical addresses to NVM cells
        
        # Statistics
        self.stats = {
            'dram_reads': 0,
            'dram_writes': 0,
            'nvm_reads': 0,
            'nvm_writes': 0,
            'dram_misses': 0,
            'page_migrations': 0,
            'nvm_wear_levels': defaultdict(int),  # Address -> wear level
        }
        
        # Current simulation time (in arbitrary units)
        self.current_time = 0
        
        # Access patterns for wear-leveling
        self.access_frequency = defaultdict(int)  # Address -> access count
        
        # Initialize empty memory cells
        self._initialize_memory()
        
    def _initialize_memory(self):
        """Initialize the DRAM and NVM with empty cells"""
        # Initialize DRAM cells
        for addr in range(self.dram_page_count):
            self.dram[addr] = DRAMCell(addr)
            
        # Initialize NVM cells
        for addr in range(self.nvm_page_count):
            self.nvm[addr] = NVMCell(addr)
    
    def allocate(self, size_bytes, prefer_nvm=False):
        """Allocate memory of specified size, return start address"""
        # Simple allocation strategy: just get the next available pages
        pages_needed = (size_bytes + self.page_size - 1) // self.page_size
        
        # Determine where to allocate (DRAM or NVM) based on preference and availability
        if prefer_nvm and len(self.address_to_nvm) + pages_needed <= self.nvm_page_count:
            # Allocate in NVM
            start_addr = len(self.address_to_nvm)
            for i in range(pages_needed):
                logical_addr = start_addr + i
                self.address_to_nvm[logical_addr] = i
            return start_addr
        else:
            # Allocate in DRAM
            start_addr = len(self.address_to_dram)
            for i in range(pages_needed):
                logical_addr = start_addr + i
                self.address_to_dram[logical_addr] = i
            return start_addr
    
    def read(self, address):
        """Read data from the memory system"""
        self.current_time += 1
        self.access_frequency[address] += 1
        
        # Check if address is in DRAM
        if address in self.address_to_dram:
            physical_addr = self.address_to_dram[address]
            cell = self.dram[physical_addr]
            cell.last_access_time = self.current_time
            self.stats['dram_reads'] += 1
            return cell.data
        
        # Check if address is in NVM
        elif address in self.address_to_nvm:
            physical_addr = self.address_to_nvm[address]
            cell = self.nvm[physical_addr]
            cell.last_access_time = self.current_time
            self.stats['nvm_reads'] += 1
            
            # Migration policy: move frequently accessed NVM data to DRAM
            if self.should_migrate_to_dram(address):
                self.migrate_to_dram(address)
                
            return cell.data
        
        # Address not found
        else:
            self.stats['dram_misses'] += 1
            return None
            
    def write(self, address, data):
        """Write data to the memory system"""
        self.current_time += 1
        self.access_frequency[address] += 1
        
        # Check if address is in DRAM
        if address in self.address_to_dram:
            physical_addr = self.address_to_dram[address]
            cell = self.dram[physical_addr]
            cell.data = data
            cell.last_access_time = self.current_time
            self.stats['dram_writes'] += 1
            return True
        
        # Check if address is in NVM
        elif address in self.address_to_nvm:
            physical_addr = self.address_to_nvm[address]
            cell = self.nvm[physical_addr]
            
            # Use NVM's write method which tracks write counts
            did_write = cell.write(data)
            if did_write:
                self.stats['nvm_writes'] += 1
            
            cell.last_access_time = self.current_time
            
            # Update wear level statistics
            self.stats['nvm_wear_levels'][address] = cell.get_wear_level()
            
            # Check if this cell is getting worn out and needs to be re-mapped
            if cell.get_wear_level() > 70:  # 70% worn
                self.wear_leveling(address)
            
            return True
        
        # Address not found
        else:
            return False
            
    def should_migrate_to_dram(self, address, threshold=10):
        """Decide if NVM data should be migrated to DRAM based on access frequency"""
        return self.access_frequency[address] > threshold
    
    def migrate_to_dram(self, address):
        """Move data from NVM to DRAM"""
        if address not in self.address_to_nvm:
            return False
        
        # Get data from NVM
        nvm_physical_addr = self.address_to_nvm[address]
        data = self.nvm[nvm_physical_addr].data
        
        # Find space in DRAM (simplified: just use the next available slot)
        dram_physical_addr = len(self.address_to_dram)
        
        # Check if DRAM has space
        if dram_physical_addr >= self.dram_page_count:
            # DRAM is full, need to evict something
            evict_addr = self.find_lru_dram_page()
            dram_physical_addr = self.address_to_dram[evict_addr]
            del self.address_to_dram[evict_addr]
        
        # Move data to DRAM
        self.dram[dram_physical_addr].data = data
        self.address_to_dram[address] = dram_physical_addr
        
        # Remove mapping from NVM (but keep the data for persistence)
        del self.address_to_nvm[address]
        
        self.stats['page_migrations'] += 1
        return True
    
    def find_lru_dram_page(self):
        """Find the least recently used page in DRAM for eviction"""
        min_time = float('inf')
        lru_addr = None
        
        for addr, physical_addr in self.address_to_dram.items():
            if self.dram[physical_addr].last_access_time < min_time:
                min_time = self.dram[physical_addr].last_access_time
                lru_addr = addr
                
        return lru_addr
    
    def wear_leveling(self, address):
        """Implement wear-leveling for NVM cells"""
        if address not in self.address_to_nvm:
            return False
        
        current_physical_addr = self.address_to_nvm[address]
        current_cell = self.nvm[current_physical_addr]
        
        # Find a less-worn cell to swap with
        target_physical_addr = None
        min_writes = current_cell.write_count
        
        for addr, cell in self.nvm.items():
            if cell.write_count < min_writes and addr != current_physical_addr:
                min_writes = cell.write_count
                target_physical_addr = addr
        
        if target_physical_addr is not None:
            # Find the logical address that points to the target physical address
            target_logical_addr = None
            for logical_addr, phys_addr in self.address_to_nvm.items():
                if phys_addr == target_physical_addr:
                    target_logical_addr = logical_addr
                    break
            
            if target_logical_addr is not None:
                # Swap the physical addresses
                self.address_to_nvm[address] = target_physical_addr
                self.address_to_nvm[target_logical_addr] = current_physical_addr
                
                # Swap the data
                temp_data = self.nvm[current_physical_addr].data
                self.nvm[current_physical_addr].data = self.nvm[target_physical_addr].data
                self.nvm[target_physical_addr].data = temp_data
                
                return True
        
        return False
    
    def refresh_dram(self):
        """Refresh all DRAM cells that need refreshing"""
        refresh_count = 0
        for addr, cell in self.dram.items():
            if cell.needs_refresh(self.current_time):
                cell.refresh(self.current_time)
                refresh_count += 1
        return refresh_count
    
    def get_memory_state(self):
        """Return the current state of memory for analysis"""
        state = {
            'dram_usage': len(self.address_to_dram) / self.dram_page_count * 100,
            'nvm_usage': len(self.address_to_nvm) / self.nvm_page_count * 100,
            'nvm_wear_avg': sum(self.stats['nvm_wear_levels'].values()) / max(1, len(self.stats['nvm_wear_levels'])),
            'nvm_wear_max': max(self.stats['nvm_wear_levels'].values()) if self.stats['nvm_wear_levels'] else 0,
            'dram_reads': self.stats['dram_reads'],
            'dram_writes': self.stats['dram_writes'],
            'nvm_reads': self.stats['nvm_reads'],
            'nvm_writes': self.stats['nvm_writes'],
            'page_migrations': self.stats['page_migrations'],
        }
        return state

class WorkloadGenerator:
    """Generates memory access patterns to simulate various workloads"""
    def __init__(self, address_space_size, seed=None):
        self.address_space_size = address_space_size
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_random_workload(self, num_operations, write_ratio=0.3):
        """Generate a random workload with given write ratio"""
        workload = []
        for _ in range(num_operations):
            is_write = random.random() < write_ratio
            address = random.randint(0, self.address_space_size - 1)
            data = random.randint(0, 255) if is_write else None
            workload.append(('write' if is_write else 'read', address, data))
        return workload
    
    def generate_sequential_workload(self, num_operations, write_ratio=0.3):
        """Generate a sequential access workload"""
        workload = []
        for i in range(num_operations):
            is_write = random.random() < write_ratio
            address = i % self.address_space_size
            data = random.randint(0, 255) if is_write else None
            workload.append(('write' if is_write else 'read', address, data))
        return workload
    
    def generate_zipfian_workload(self, num_operations, alpha=1.0, write_ratio=0.3):
        """Generate a Zipfian (skewed) workload - some addresses accessed much more than others"""
        # Generate Zipfian distribution
        x = np.arange(1, self.address_space_size + 1)
        weights = x ** (-alpha)
        weights /= weights.sum()
        
        workload = []
        addresses = np.random.choice(
            np.arange(self.address_space_size), 
            size=num_operations, 
            p=weights
        )
        
        for addr in addresses:
            is_write = random.random() < write_ratio
            data = random.randint(0, 255) if is_write else None
            workload.append(('write' if is_write else 'read', addr, data))
            
        return workload
    
    def generate_loop_workload(self, num_operations, loop_size=100, write_ratio=0.3):
        """Generate a workload that simulates a loop accessing a fixed memory region"""
        workload = []
        for i in range(num_operations):
            is_write = random.random() < write_ratio
            # Loop over a fixed memory region
            address = i % loop_size
            data = random.randint(0, 255) if is_write else None
            workload.append(('write' if is_write else 'read', address, data))
        return workload

class HybridMemorySimulation:
    """Main simulation class that runs workloads and collects statistics"""
    def __init__(self, dram_size_mb=256, nvm_size_mb=1024, page_size_kb=4):
        self.memory_controller = MemoryController(
            dram_size_mb=dram_size_mb,
            nvm_size_mb=nvm_size_mb,
            page_size_kb=page_size_kb
        )
        
        # Total address space size (in pages)
        total_pages = self.memory_controller.dram_page_count + self.memory_controller.nvm_page_count
        self.workload_generator = WorkloadGenerator(total_pages)
        
        # Results storage
        self.results = []
        
    def run_workload(self, workload, description=""):
        """Run a given workload and collect statistics"""
        start_time = time.time()
        
        # Get initial memory state
        initial_state = self.memory_controller.get_memory_state()
        
        # Execute each operation in the workload
        for op_type, address, data in workload:
            if op_type == 'read':
                self.memory_controller.read(address)
            elif op_type == 'write':
                self.memory_controller.write(address, data)
                
            # Periodically refresh DRAM
            if self.memory_controller.current_time % 100 == 0:
                self.memory_controller.refresh_dram()
        
        # Get final memory state
        final_state = self.memory_controller.get_memory_state()
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Store results
        result = {
            'description': description,
            'workload_size': len(workload),
            'elapsed_time': elapsed_time,
            'initial_state': initial_state,
            'final_state': final_state,
            'operations_per_second': len(workload) / elapsed_time,
            'dram_reads': self.memory_controller.stats['dram_reads'],
            'dram_writes': self.memory_controller.stats['dram_writes'],
            'nvm_reads': self.memory_controller.stats['nvm_reads'],
            'nvm_writes': self.memory_controller.stats['nvm_writes'],
            'page_migrations': self.memory_controller.stats['page_migrations'],
            'dram_miss_rate': self.memory_controller.stats['dram_misses'] / max(1, len(workload)),
            'nvm_wear_levels': dict(self.memory_controller.stats['nvm_wear_levels']),
        }
        
        self.results.append(result)
        return result
    
    def reset_stats(self):
        """Reset the memory controller statistics"""
        self.memory_controller.stats = {
            'dram_reads': 0,
            'dram_writes': 0,
            'nvm_reads': 0,
            'nvm_writes': 0,
            'dram_misses': 0,
            'page_migrations': 0,
            'nvm_wear_levels': defaultdict(int),
        }
    
    def compare_workloads(self, workloads_dict):
        """Run multiple workloads and compare their performance"""
        comparison_results = []
        
        for name, workload in workloads_dict.items():
            self.reset_stats()
            result = self.run_workload(workload, description=name)
            comparison_results.append(result)
            
        return comparison_results
    
    def visualize_results(self, results=None, save_dir="results"):
        """Visualize simulation results"""
        if results is None:
            results = self.results
            
        if not results:
            print("No results to visualize")
            return
            
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a DataFrame from results for easier manipulation
        df = pd.DataFrame(results)
        
        # 1. Performance Comparison: Operations per Second
        plt.figure(figsize=(10, 6))
        plt.bar(df['description'], df['operations_per_second'])
        plt.title('Performance Comparison: Operations per Second')
        plt.xlabel('Workload')
        plt.ylabel('Operations per Second')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/performance_comparison_{timestamp}.png")
        
        # 2. Memory Access Distribution
        plt.figure(figsize=(12, 8))
        for i, result in enumerate(results):
            reads = [result['dram_reads'], result['nvm_reads']]
            writes = [result['dram_writes'], result['nvm_writes']]
            
            x = np.arange(2)
            width = 0.35
            
            plt.bar(x - width/2 + i*width/len(results), reads, width/len(results), label=f'{result["description"]} Reads')
            plt.bar(x + width/2 + i*width/len(results), writes, width/len(results), label=f'{result["description"]} Writes', alpha=0.7)
        
        plt.title('Memory Access Distribution')
        plt.xlabel('Memory Type')
        plt.ylabel('Number of Accesses')
        plt.xticks(x, ['DRAM', 'NVM'])
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/memory_access_distribution_{timestamp}.png")
        
        # 3. NVM Wear Levels
        plt.figure(figsize=(10, 6))
        for result in results:
            wear_levels = list(result['nvm_wear_levels'].values())
            if wear_levels:
                plt.hist(wear_levels, bins=20, alpha=0.7, label=result['description'])
        
        plt.title('NVM Wear Level Distribution')
        plt.xlabel('Wear Level (%)')
        plt.ylabel('Number of Cells')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/nvm_wear_distribution_{timestamp}.png")
        
        # 4. Page Migration Analysis
        plt.figure(figsize=(10, 6))
        plt.bar(df['description'], df['page_migrations'])
        plt.title('Page Migrations: DRAM <-> NVM')
        plt.xlabel('Workload')
        plt.ylabel('Number of Migrations')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/page_migrations_{timestamp}.png")
        
        # 5. Summary Table
        summary = df[['description', 'operations_per_second', 'dram_miss_rate', 'page_migrations']]
        summary['nvm_wear_avg'] = df.apply(
            lambda row: sum(row['nvm_wear_levels'].values()) / max(1, len(row['nvm_wear_levels'])), 
            axis=1
        )
        
        # Save summary to CSV
        summary.to_csv(f"{save_dir}/summary_{timestamp}.csv", index=False)
        
        print(f"Visualizations and summary saved to {save_dir}/ directory")
        return summary

def run_simulation_experiment():
    """Run a comprehensive simulation experiment with different workloads"""
    print("Starting Hybrid Memory Simulation Experiment...")
    
    # Initialize simulation
    sim = HybridMemorySimulation(dram_size_mb=256, nvm_size_mb=1024)
    
    # Define workloads to test
    workloads = {}
    
    # 1. Random workload
    print("Generating random workload...")
    workloads["Random Access"] = sim.workload_generator.generate_random_workload(
        num_operations=100000, 
        write_ratio=0.3
    )
    
    # 2. Sequential workload
    print("Generating sequential workload...")
    workloads["Sequential Access"] = sim.workload_generator.generate_sequential_workload(
        num_operations=100000, 
        write_ratio=0.3
    )
    
    # 3. Zipfian (skewed) workload
    print("Generating zipfian workload...")
    workloads["Zipfian Access"] = sim.workload_generator.generate_zipfian_workload(
        num_operations=100000, 
        write_ratio=0.3
    )
    
    # 4. Loop workload (to test wear leveling)
    print("Generating loop workload...")
    workloads["Loop Access"] = sim.workload_generator.generate_loop_workload(
        num_operations=100000, 
        loop_size=100,
        write_ratio=0.7  # Higher write ratio to stress test wear leveling
    )
    
    # Run comparison
    print("Running workload comparison...")
    results = sim.compare_workloads(workloads)
    
    # Visualize results
    print("Visualizing results...")
    summary = sim.visualize_results()
    
    print("\nExperiment Summary:")
    print(summary)
    
    return sim, results

if __name__ == "__main__":
    sim, results = run_simulation_experiment()