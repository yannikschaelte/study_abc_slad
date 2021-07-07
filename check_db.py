import pyabc
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--db", type=str)
args = parser.parse_args()
db = args.db

print(sum(pyabc.History("sqlite:///" + db, create=False).get_all_populations()['samples']))
