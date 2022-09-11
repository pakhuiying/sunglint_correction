import argparse

parser = argparse.ArgumentParser(description='GUI sunglint selection')

parser.add_argument('--prefix',type=str,default = 'test',help='prefix for filename')

parser.add_argument('--fp_store',type=str,help='directory where images are stored')

parser.add_argument('--fp_save',type=str, default='/json_files',help='directory to save the files')

args = parser.parse_args()

