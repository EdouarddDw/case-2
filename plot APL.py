import matplotlib.pyplot as plt
import pandas as pd

try:
    apl_df = pd.read_csv('top_20_apl_candidates.csv')
except FileNotFoundError:
    print("top_20_apl_candidates.csv not found – skipping APL chart.")
else:
    apl_df = apl_df.sort_values('net_benefit', ascending=False)

    plt.figure(figsize=(12, 6))
    plt.bar(apl_df['node_id'].astype(str), apl_df['net_benefit'])
    plt.title('Annual Net Benefit per Proposed APL (after €10k cost)')
    plt.xlabel('Node ID')
    plt.ylabel('Net Benefit (€)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('apl_net_benefit_bar.png', dpi=300)
    plt.show()
    print("APL bar chart saved as apl_net_benefit_bar.png")