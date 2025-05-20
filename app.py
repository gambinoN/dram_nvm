import streamlit as st
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import your simulation classes
from hybrid_memory_simulation import HybridMemorySimulation, WorkloadGenerator

# --- Streamlit App ---

def main():
    st.set_page_config(page_title="Hybrid Memory Simulator", layout="wide")
    st.title("Hybrid DRAM + NVM Simulator")

    # Sidebar controls
    with st.sidebar:
        st.header("Simulation Parameters")
        dram_size_mb = st.number_input("DRAM size (MiB)", min_value=1, value=256)
        nvm_size_mb = st.number_input("NVM size (MiB)", min_value=1, value=1024)
        page_size_kb = st.number_input("Page size (KiB)", min_value=1, value=4)
        num_ops = st.number_input("Number of operations", min_value=1, value=100000)
        write_ratio = st.slider("Write ratio", 0.0, 1.0, 0.3)
        seed = st.number_input("Random seed", min_value=0, value=42)
        workloads = st.multiselect(
            "Workloads to run",
            options=["random", "sequential", "zipfian", "loop"],
            default=["random", "sequential", "zipfian", "loop"]
        )
        loop_size = st.number_input("Loop size (for loop workload)", min_value=1, value=100)
        run_button = st.button("Run Simulation")

    if run_button:
        # Set seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)

        # Initialize simulation
        sim = HybridMemorySimulation(
            dram_size_mb=dram_size_mb,
            nvm_size_mb=nvm_size_mb,
            page_size_kb=page_size_kb
        )

        # Build selected workloads
        workloads_dict = {}
        if "random" in workloads:
            workloads_dict["Random"] = sim.gen.generate_random_workload(num_ops, write_ratio)
        if "sequential" in workloads:
            workloads_dict["Sequential"] = sim.gen.generate_sequential_workload(num_ops, write_ratio)
        if "zipfian" in workloads:
            workloads_dict["Zipfian"] = sim.gen.generate_zipfian_workload(num_ops, write_ratio)
        if "loop" in workloads:
            workloads_dict["Loop"] = sim.gen.generate_loop_workload(num_ops, loop_size, write_ratio)

        # Run simulation
        with st.spinner("Running workloads… this may take a moment"):
            results = sim.compare_workloads(workloads_dict)

        # Convert results to DataFrame
        df = pd.DataFrame(results)

        # Display summary table
        st.subheader("Simulation Summary")
        st.dataframe(df)

        # Charts
        st.subheader("Operations per Second by Workload")
        st.bar_chart(df.set_index("description")["ops_per_sec"])

        st.subheader("Page Migrations by Workload")
        st.bar_chart(df.set_index("description")["page_migrations"])

        st.subheader("DRAM Miss Rate by Workload")
        st.bar_chart(df.set_index("description")["dram_miss_rate"])

        st.subheader("NVM Wear Level Distribution (All Workloads)")
        all_wear = []
        for res in results:
            all_wear.extend(res['nvm_wear_levels'].values())
        if all_wear:
            fig, ax = plt.subplots()
            ax.hist(all_wear, bins=20)
            ax.set_xlabel('Wear Level (%)')
            ax.set_ylabel('Number of Cells')
            st.pyplot(fig)
        else:
            st.write("No NVM writes recorded; wear distribution is empty.")

if __name__ == '__main__':
    main()
