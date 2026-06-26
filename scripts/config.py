import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--subtype', required=True)
    parser.add_argument('--status_col', default='Disease_Status')
    parser.add_argument('--pathway_options', type=lambda s: s.split(','), default=['gobp', 'reactome', 'HALLMARK-KEGG'])
    return parser.parse_args()

def make_output_path(base_dir, subtype):
    output_path = os.path.join(base_dir, subtype)
    os.makedirs(output_path, exist_ok=True)
    return output_path
